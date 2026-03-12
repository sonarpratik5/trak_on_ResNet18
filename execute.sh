#!/bin/bash
set -e

echo "Starting ResNet-TRAK Pipeline..."

# 1. Walk back into the correct Docker directory
cd /app

# 2. Run your code
python main.py --mode all

# 3. Save the results to your permanent team drive before the container dies
TEAM_DIR="/home/exml_team013/trak/output_data"
mkdir -p $TEAM_DIR

cp -r /app/checkpoints $TEAM_DIR/
cp -r /app/trak_results $TEAM_DIR/
cp /app/gradcam_analysis.png $TEAM_DIR/
cp /app/attribution_scores.npy $TEAM_DIR/

echo "Pipeline Complete and Data Saved!"
