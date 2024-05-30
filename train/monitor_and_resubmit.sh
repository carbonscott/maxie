#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <JOB_NAME> <SCRIPT_PATH> <PENDING_CHECK_INTERVAL; in sec> <RUNNING_CHECK_INTERVAL; in sec> <RESUB_DELAY_TIME; in sec>"
    exit 1
fi

JOB_NAME=$1
SCRIPT_PATH=$2
PENDING_CHECK_INTERVAL=$3  # seconds
RUNNING_CHECK_INTERVAL=$4  # seconds
RESUB_DELAY_TIME=$5        # seconds

# Function to get job ID by job name
get_job_id() {
    bjobs -J "$JOB_NAME" -noheader -o "jobid" 2>&1 | head -n 1
}

# Function to get job status
get_job_status() {
    local job_id=$1
    bjobs -noheader -o "stat" "$job_id" 2>&1
}

# Function to resubmit the job
resubmit_job() {
    echo "Resubmitting the job in $((RESUB_DELAY_TIME)) seconds."
    sleep $RESUB_DELAY_TIME
    bsub "$SCRIPT_PATH"
    if [ $? -eq 0 ]; then
        echo "Job resubmitted successfully."
    else
        echo "Failed to resubmit job."
    fi
}

# Main monitoring loop
while true; do
    JOB_ID=$(get_job_id)

    if [[ "$JOB_ID" == *"not found"* ]]; then
        echo "Job $JOB_NAME not found."
        sleep $PENDING_CHECK_INTERVAL
        continue
    fi

    echo "Monitoring Job $JOB_NAME with ID $JOB_ID."

    while true; do
        JOB_STATUS=$(get_job_status "$JOB_ID")

        if [[ "$JOB_STATUS" == *"not found"* ]]; then
            echo "Job $JOB_ID not found. This likely means the job has completed or was terminated."
            resubmit_job
            break
        fi

        echo "Job $JOB_ID status: $JOB_STATUS"

        if [[ "$JOB_STATUS" == "PEND" ]]; then
            echo "Job is pending. Checking again in $PENDING_CHECK_INTERVAL seconds."
            sleep $PENDING_CHECK_INTERVAL
        elif [[ "$JOB_STATUS" == "RUN" ]]; then
            echo "Job is running. Checking again in $((RUNNING_CHECK_INTERVAL)) seconds."
            while true; do
                sleep $RUNNING_CHECK_INTERVAL
                JOB_STATUS=$(get_job_status "$JOB_ID")
                if [[ "$JOB_STATUS" != "RUN" ]]; then
                    echo "Job $JOB_ID has completed or is no longer running. Resubmitting..."
                    resubmit_job
                    break 2
                fi
                echo "Job $JOB_ID is still running. Checking again in $((RUNNING_CHECK_INTERVAL)) seconds."
            done
        else
            echo "Unexpected job status: $JOB_STATUS. Exiting."
            exit 1
        fi
    done
done
