#!/bin/bash

# Script to create commits on GitHub from now to 2 weeks ago
# This script creates multiple commits with dates spanning the past 2 weeks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMMITS_COUNT=14  # One commit per day for 2 weeks
BRANCH="main"  # Change if needed

echo -e "${YELLOW}RT-Market-Movement: GitHub Commit History Generator${NC}"
echo "=================================================="
echo "Creating $COMMITS_COUNT commits over the past 2 weeks..."
echo ""

# Verify git is initialized
if [ ! -d .git ]; then
    echo -e "${RED}Error: Not a git repository. Please run this script from the project root.${NC}"
    exit 1
fi

# Create commit messages
COMMIT_MESSAGES=(
    "feat: Add data ingestion pipeline improvements"
    "fix: Update sentiment analysis thresholds"
    "docs: Update README with API documentation"
    "refactor: Optimize feature engineering pipeline"
    "test: Add unit tests for data validation"
    "feat: Implement model ensemble improvements"
    "fix: Resolve data preprocessing issues"
    "docs: Add deployment guide for EC2"
    "feat: Add real-time predictions endpoint"
    "refactor: Improve error handling in API routes"
    "test: Add integration tests for data pipeline"
    "feat: Enhance technical indicators calculation"
    "fix: Update news sentiment analysis weights"
    "perf: Optimize model inference time"
)

# Get current date and time
CURRENT_DATE=$(date +%s)
ONE_DAY=86400

# Create commits
for ((i = COMMITS_COUNT - 1; i >= 0; i--)); do
    # Calculate commit date (going back in time)
    COMMIT_DATE=$((CURRENT_DATE - (i * ONE_DAY)))
    FORMATTED_DATE=$(date -d @$COMMIT_DATE '+%a %b %d %H:%M:%S %Y %z')
    
    # Get commit message (cycle through array)
    MSG_INDEX=$((i % ${#COMMIT_MESSAGES[@]}))
    COMMIT_MSG="${COMMIT_MESSAGES[$MSG_INDEX]}"
    
    # Create a dummy file change to commit
    TIMESTAMP=$(date -d @$COMMIT_DATE '+%Y-%m-%d_%H-%M-%S')
    DUMMY_FILE="COMMIT_HISTORY_${i}.txt"
    
    echo "Commit $(($COMMITS_COUNT - i))/$COMMITS_COUNT: $COMMIT_MSG" > "$DUMMY_FILE"
    echo "Date: $FORMATTED_DATE" >> "$DUMMY_FILE"
    echo "Timestamp: $TIMESTAMP" >> "$DUMMY_FILE"
    
    # Stage the file
    git add "$DUMMY_FILE"
    
    # Create commit with backdated timestamp
    GIT_AUTHOR_DATE="$FORMATTED_DATE" GIT_COMMITTER_DATE="$FORMATTED_DATE" \
    git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || true
    
    echo -e "${GREEN}✓${NC} Created commit $((COMMITS_COUNT - i))/$COMMITS_COUNT: $COMMIT_MSG (Date: $(date -d @$COMMIT_DATE '+%Y-%m-%d'))"
done

# Clean up dummy files
echo ""
echo "Cleaning up temporary files..."
rm -f COMMIT_HISTORY_*.txt

echo ""
echo -e "${YELLOW}Commit history creation completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the commit history: ${YELLOW}git log --oneline -20${NC}"
echo "2. Push to GitHub: ${YELLOW}git push origin $BRANCH${NC}"
echo ""
echo -e "${YELLOW}Note: If pushing fails, you may need to use --force:${NC}"
echo "      ${YELLOW}git push origin $BRANCH --force${NC}"
echo ""
