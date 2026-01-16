#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Traffic Simulation Evolution Runner ===${NC}"
echo ""

read -rp "Enter number of times to run Evolution Loop: " num
read -rp "Display simulation at end? [Y/N]: " display

if [[ "$num" =~ ^[0-9]+$ ]]; then
    echo ""
    echo -e "${GREEN}Running Evolution Loop $num times${NC}"
    echo "Started at: $(date)"
    echo ""

    for ((i=1; i<=num; i++)); do
        echo -e "${YELLOW}--- Run $i of $num ---${NC}"
        cargo run -q --release -p evolution_main
        echo ""
    done

    echo -e "${GREEN}Evolution complete!${NC}"
    echo "Finished at: $(date)"
    echo ""

    # Always generate visualizations
    echo -e "${BLUE}Generating visualizations...${NC}"

    # Check if Python is available
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "Error: Python not found. Please install Python to generate visualizations."
        PYTHON_CMD=""
    fi

    if [[ -n "$PYTHON_CMD" ]]; then
        # Install requirements if needed (silently)
        if [[ -f "visualization/requirements.txt" ]]; then
            $PYTHON_CMD -m pip install -q -r visualization/requirements.txt 2>/dev/null
        fi

        # Run consolidated visualization script
        if [[ -f "visualization/visualize.py" ]]; then
            echo -e "${YELLOW}Running visualization...${NC}"
            $PYTHON_CMD visualization/visualize.py

            echo ""
            echo -e "${GREEN}All visualizations saved to output/serialization/graphs/${NC}"
        else
            echo "Warning: visualization/visualize.py not found"
        fi
    fi
    echo ""

    # Display simulation if requested
    if [[ "${display^^}" == "Y" ]]; then
        echo -e "${BLUE}Launching simulation display...${NC}"
        cargo run --release -p draw_main
    fi

else
    echo "Error: input must be a positive integer"
    exit 1
fi
