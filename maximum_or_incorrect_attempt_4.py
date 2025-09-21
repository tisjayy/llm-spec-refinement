class Solution:
    def maximumOr(self, nums, k):
        total_or = 0
        for num in nums:
            total_or |= num
        
        max_or = total_or
        
        for i in range(len(nums)):
            original_number = nums[i]
            modified_number = original_number * (2 ** k)
            new_or = total_or ^ original_number | modified_number
            max_or = max(max_or, new_or)
        
        return max_or
