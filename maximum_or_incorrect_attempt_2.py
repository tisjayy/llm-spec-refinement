from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        max_or = 0
        
        # Calculate the initial OR of the array
        initial_or = 0
        for num in nums:
            initial_or |= num
        
        # Try multiplying each number by 2 up to k times
        for i in range(len(nums)):
            current_num = nums[i]
            for j in range(k + 1):
                # Calculate the OR if we multiply current_num by 2^j
                modified_num = current_num * (2 ** j)
                current_or = initial_or ^ current_num | modified_num
                max_or = max(max_or, current_or)
        
        return max(max_or, initial_or)