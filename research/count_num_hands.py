from functools import lru_cache

def count_unique_hands():
    # Limits for each rank.
    # 11 ranks (3 through K) have 4 copies,
    # Ace has 3 copies, and 2 has 1 copy.
    limits = [4] * 11 + [3, 1]
    total_ranks = len(limits)  # Should be 13

    @lru_cache(maxsize=None)
    def dp(i, remaining):
        """Count the ways to choose 'remaining' cards from ranks i..total_ranks-1."""
        # If we've processed all ranks, success only if no cards remain.
        if i == total_ranks:
            return 1 if remaining == 0 else 0
        
        ways = 0
        # For the current rank i, choose x cards, where 0 <= x <= min(limit, remaining)
        for x in range(0, min(limits[i], remaining) + 1):
            ways += dp(i + 1, remaining - x)
        return ways

    return dp(0, 16)

if __name__ == '__main__':
    num_hands = count_unique_hands()
    print("Number of unique starting hands:", num_hands)
