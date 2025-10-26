import os
import json
import logging
from tqdm import tqdm

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.prompts.self_repair import format_prompt_self_repair, extract_code

# --- 1. Import the correct, existing evaluation function ---
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations_by_problem

# --- 2. Import our RL components ---
from lcb_runner.runner.rl_agent import RLAgent

# --- Define Action Space and Map Actions to Strategies ---
PROMPT_ACTIONS = {
    0: "default",
    1: "step_by_step_reasoning",
    2: "error_specific_feedback",
    3: "syntax_check",
    4: "edge_case_check",
}
ACTION_DIM = len(PROMPT_ACTIONS)

class RLSelfRepairRunner:
    """
    A dedicated runner for the self-repair scenario that uses Reinforcement Learning.
    This runner is called by `build_runner` when --scenario is selfrepair and --use_rl is active.
    """
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model
        self._base_runner = None  # Lazy initialization
        
        self.rl_agent = RLAgent(action_dim=ACTION_DIM)
        self.max_repair_attempts = getattr(args, 'max_repair_attempts', 5)

        # Construct the path to the base code generation results file
        codegen_eval_path = f"output/{self.model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
        if not os.path.exists(codegen_eval_path):
            raise FileNotFoundError(f"Base codegen results not found at {codegen_eval_path}. Please run a codegen scenario first.")
        
        logging.info(f"Loading base codegen results from: {codegen_eval_path}")
        with open(codegen_eval_path, 'r') as f:
            self.codegen_results_map = {res['question_id']: res for res in json.load(f)}

    @property
    def base_runner(self):
        """Lazy initialization of the base runner to avoid circular dependencies."""
        if self._base_runner is None:
            from lcb_runner.runner.runner_utils import build_runner
            
            # Build a temporary base runner (not the RL one) for making model calls
            # Temporarily disable use_rl to get the normal runner
            original_use_rl = self.args.use_rl
            self.args.use_rl = False
            try:
                self._base_runner = build_runner(self.args, self.model)
            finally:
                self.args.use_rl = original_use_rl
        return self._base_runner

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        """
        Main execution method called by `eval_benchmark.py`.
        It runs the RL loop for each problem and returns the final raw model outputs.
        """
        all_final_outputs = []
        for sample in tqdm(benchmark, desc="RL Self-Repair Runner"):
            if sample.question_id not in self.codegen_results_map:
                logging.warning(f"Could not find base result for {sample.question_id}. Skipping.")
                all_final_outputs.append([""]) # Append empty string if no base code
                continue
            
            base_result = self.codegen_results_map[sample.question_id]
            final_model_output = self.run_sample_episode(sample, base_result, format_prompt)
            all_final_outputs.append([final_model_output])
            
        return all_final_outputs

    def run_sample_episode(self, sample, base_result, format_prompt) -> str:
        """
        Runs a full RL episode for a single problem and returns the final raw model output.
        """
        # We only try to repair the first generated code sample (as n=1 is supported for self-repair)
        code = base_result["code_list"][0]
        passed = base_result["graded_list"][0]  # Changed from passed_list to graded_list
        metadata_str = base_result["metadata"][0] if isinstance(base_result["metadata"], list) else base_result["metadata"]
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        original_model_output = base_result["output_list"][0]

        if passed:
            return original_model_output

        self.rl_agent.clear_memory()
        
        current_code = code
        current_metadata = metadata
        # We need the test case results from the initial attempt to calculate the first partial reward
        # This requires re-evaluating the initial code.
        eval_args = ([current_code], sample.get_evaluation_sample(), self.args.debug, self.args.timeout)
        current_test_results, _ = evaluate_generations_by_problem(eval_args)
        current_test_results = current_test_results[0]
        
        final_model_output = original_model_output

        attempt_pbar = tqdm(range(self.max_repair_attempts), desc="Repair attempts", leave=False)
        for attempt in attempt_pbar:
            state = {"question": sample.question_content, "code": current_code, "metadata": current_metadata}
            action_idx = self.rl_agent.select_action(state)
            prompt_strategy = PROMPT_ACTIONS[action_idx]
            
            attempt_pbar.set_description(f"Attempt {attempt+1}/{self.max_repair_attempts}: {prompt_strategy}")
            
            prompt = format_prompt(
                question=sample.question_content, LanguageModelStyle=self.model.model_style,
                code=current_code, result=False, metadata=json.dumps(current_metadata) if isinstance(current_metadata, dict) else current_metadata, strategy=prompt_strategy
            )
            
            # Use the base runner's prompts_to_outputs method to generate
            outputs = self.base_runner.prompts_to_outputs([prompt])
            model_output = outputs[0][0]  # Get the first (and only) output
            final_model_output = model_output
            
            repaired_code = extract_code(model_output, self.model.model_style)
            
            # --- 3. Call the correct evaluation function ---
            eval_args = ([repaired_code], sample.get_evaluation_sample(), self.args.debug, self.args.timeout)
            new_eval_results, new_metadata_list = evaluate_generations_by_problem(eval_args)
            new_test_results = new_eval_results[0]
            new_metadata = new_metadata_list[0]
            new_passed = all(new_test_results)

            # --- 4. Compute reward using the list of test case results ---
            reward = self.compute_reward_from_test_cases(new_test_results, current_test_results)
            self.rl_agent.store_reward(reward)
            
            # Update progress bar with results
            passed_tests = sum(new_test_results)
            total_tests = len(new_test_results)
            attempt_pbar.set_postfix({"passed": f"{passed_tests}/{total_tests}", "reward": f"{reward:.2f}"})

            # Update state for the next iteration
            current_code = repaired_code
            current_metadata = new_metadata
            current_test_results = new_test_results # CRITICAL: Update for the next loop
            
            if new_passed:
                attempt_pbar.set_description("✓ Repair successful!")
                attempt_pbar.close()
                break
        
        self.rl_agent.update_policy()
        return final_model_output

    def compute_reward_from_test_cases(self, new_test_results: list, old_test_results: list) -> float:
        """
        Computes a reward based on the change in passed test cases.
        """
        if all(new_test_results):
            return 1.0  # Max reward for full success

        new_passed_count = sum(1 for r in new_test_results if r is True)
        old_passed_count = sum(1 for r in old_test_results if r is True)
        total_count = len(new_test_results)

        if total_count == 0:
            return 0.0 # Avoid division by zero

        # Give a positive reward for any incremental progress
        if new_passed_count > old_passed_count:
            progress = (new_passed_count - old_passed_count) / total_count
            return progress
        
        # Give a negative reward (penalty) if progress is stagnant or regresses
        else:
            # You could add more nuanced penalties here based on error codes if desired
            return -0.5 # Penalty for no improvement or regression