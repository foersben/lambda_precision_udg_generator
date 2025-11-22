#!/bin/bash
# Local CI testing script using act
# Run this before pushing to test CI workflows locally

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Lambda Precision UDG Generator - Local CI Tests ===${NC}\n"

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo -e "${RED}Error: 'act' is not installed.${NC}"
    echo "Please install act: https://github.com/nektos/act"
    echo ""
    echo "Linux: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
    echo "macOS: brew install act"
    echo "Windows: choco install act-cli"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running.${NC}"
    echo "Please start Docker and try again."
    exit 1
fi

# Parse arguments
JOB="${1:-all}"
VERBOSE="${2:-}"

echo -e "${GREEN}Docker is running ✓${NC}"
echo -e "${GREEN}Act is installed ✓${NC}\n"

# Function to run a job
run_job() {
    local job_name=$1
    local description=$2

    echo -e "${YELLOW}Running: $description${NC}"

    if [ "$VERBOSE" = "-v" ]; then
        act -j "$job_name" -v
    else
        act -j "$job_name"
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $description passed${NC}\n"
        return 0
    else
        echo -e "${RED}✗ $description failed${NC}\n"
        return 1
    fi
}

# Main execution
case "$JOB" in
    "lint"|"lint-and-format")
        run_job "lint-and-format" "Lint and Format Check"
        ;;

    "type"|"type-check")
        run_job "type-check" "Type Check with MyPy"
        ;;

    "test")
        run_job "test" "Tests"
        ;;

    "coverage"|"test-with-coverage")
        run_job "test-with-coverage" "Tests with Coverage"
        ;;

    "quick")
        echo -e "${YELLOW}Running quick checks (lint + type)...${NC}\n"
        run_job "lint-and-format" "Lint and Format Check"
        LINT_RESULT=$?

        run_job "type-check" "Type Check with MyPy"
        TYPE_RESULT=$?

        if [ $LINT_RESULT -eq 0 ] && [ $TYPE_RESULT -eq 0 ]; then
            echo -e "${GREEN}✓ All quick checks passed!${NC}"
            exit 0
        else
            echo -e "${RED}✗ Some checks failed${NC}"
            exit 1
        fi
        ;;

    "all")
        echo -e "${YELLOW}Running all CI checks...${NC}\n"

        run_job "lint-and-format" "Lint and Format Check"
        LINT_RESULT=$?

        run_job "type-check" "Type Check with MyPy"
        TYPE_RESULT=$?

        run_job "test" "Tests"
        TEST_RESULT=$?

        run_job "test-with-coverage" "Tests with Coverage"
        COV_RESULT=$?

        echo -e "\n${YELLOW}=== Summary ===${NC}"
        [ $LINT_RESULT -eq 0 ] && echo -e "${GREEN}✓ Lint${NC}" || echo -e "${RED}✗ Lint${NC}"
        [ $TYPE_RESULT -eq 0 ] && echo -e "${GREEN}✓ Type Check${NC}" || echo -e "${RED}✗ Type Check${NC}"
        [ $TEST_RESULT -eq 0 ] && echo -e "${GREEN}✓ Tests${NC}" || echo -e "${RED}✗ Tests${NC}"
        [ $COV_RESULT -eq 0 ] && echo -e "${GREEN}✓ Coverage${NC}" || echo -e "${RED}✗ Coverage${NC}"

        if [ $LINT_RESULT -eq 0 ] && [ $TYPE_RESULT -eq 0 ] && [ $TEST_RESULT -eq 0 ] && [ $COV_RESULT -eq 0 ]; then
            echo -e "\n${GREEN}✓ All checks passed! Ready to push.${NC}"
            exit 0
        else
            echo -e "\n${RED}✗ Some checks failed. Fix issues before pushing.${NC}"
            exit 1
        fi
        ;;

    "list")
        echo "Available workflows and jobs:"
        act -l
        ;;

    "help"|"-h"|"--help")
        echo "Usage: ./test-ci.sh [JOB] [-v]"
        echo ""
        echo "Jobs:"
        echo "  lint              - Run linting and formatting checks"
        echo "  type              - Run type checking with MyPy"
        echo "  test              - Run test suite"
        echo "  coverage          - Run tests with coverage"
        echo "  quick             - Run lint + type (fast)"
        echo "  all               - Run all checks (default)"
        echo "  list              - List all available jobs"
        echo "  help              - Show this help message"
        echo ""
        echo "Options:"
        echo "  -v                - Verbose output"
        echo ""
        echo "Examples:"
        echo "  ./test-ci.sh              # Run all checks"
        echo "  ./test-ci.sh quick        # Run quick checks"
        echo "  ./test-ci.sh lint         # Run only linting"
        echo "  ./test-ci.sh test -v      # Run tests with verbose output"
        ;;

    *)
        echo -e "${RED}Unknown job: $JOB${NC}"
        echo "Run './test-ci.sh help' for usage information"
        exit 1
        ;;
esac
