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
read -rp "Generate visualization graphs at end? [Y/N]: " visualize

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

    # Generate visualization if requested
    if [[ "${visualize^^}" == "Y" ]]; then
        echo -e "${BLUE}Generating visualization...${NC}"

        # Check if Python and requirements are available
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

            # Run visualization script
            if [[ -f "metrics.csv" ]]; then
                $PYTHON_CMD visualization/plot_metrics.py metrics.csv
                echo -e "${GREEN}Visualization saved to evolution_progress.png${NC}"
            else
                echo "Warning: metrics.csv not found. No visualization generated."
            fi
        fi
        echo ""
    fi

    # Display simulation if requested
    if [[ "${display^^}" == "Y" ]]; then
        echo -e "${BLUE}Launching simulation display...${NC}"
        cargo run --release -p draw_main
    fi

else
    echo "Error: input must be a positive integer"
    exit 1
fi
