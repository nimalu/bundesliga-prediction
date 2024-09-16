#!/bin/bash

rm db.sqlite3

for notebook in ./*.ipynb; do
    START_TIME=$(date +%s)
    jupyter nbconvert --to notebook --execute "$notebook" --ExecutePreprocessor.timeout=-1 --output "$notebook"
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED_TIME / 60))
    SECONDS=$((ELAPSED_TIME % 60))
    echo "Finished in  ${MINUTES}m ${SECONDS}s"
done

echo "All notebooks executed."