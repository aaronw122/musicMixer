# A/B Test Suite — sound quality flag evaluation
# All commands run from the backend dir where uv resolves dependencies

ab-test:  ## Run A/B compare test (all pairs, control vs enhanced)
	cd backend && uv run python ../scripts/run_ab_test_suite.py --mode compare

ab-test-quick:  ## Quick smoke test (pair 1 only, compare mode)
	cd backend && uv run python ../scripts/run_ab_test_suite.py --mode compare --pairs 1

ab-sweep:  ## Run A/B sweep test (all pairs, per-flag isolation)
	cd backend && uv run python ../scripts/run_ab_test_suite.py --mode sweep

ab-sweep-quick:  ## Quick sweep test (pair 1 only)
	cd backend && uv run python ../scripts/run_ab_test_suite.py --mode sweep --pairs 1

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: ab-test ab-test-quick ab-sweep ab-sweep-quick help
.DEFAULT_GOAL := help
