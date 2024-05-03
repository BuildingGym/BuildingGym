import asyncio

async def my_task():
    while True:
        # Your task logic goes here
        print("Executing task...")
        await asyncio.sleep(5)  # Wait for 5 seconds before executing again
        # Check if a condition to stop the task is met
        # if condition_to_stop_task:
        #     break

async def main():
    # Start the task
    task = asyncio.create_task(my_task())

    # Let the task run for some time
    await asyncio.sleep(20)

    # Cancel the task
    task.cancel()

    try:
        # Wait for the task to complete
        await task
    except asyncio.CancelledError:
        print("Task was cancelled.")

# Run the main coroutine
asyncio.run(main())