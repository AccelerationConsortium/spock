import asyncio, grpc, chat_pb2, chat_pb2_grpc

async def main():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = chat_pb2_grpc.ChatStub(channel)
        req  = chat_pb2.ChatRequest(
            user_id="alice",
            prompt="Explain quantum teleportation in 3 sentences.",
            use_trt=True
        )
        async for delta in stub.StreamChat(req):
            print(delta.text, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
