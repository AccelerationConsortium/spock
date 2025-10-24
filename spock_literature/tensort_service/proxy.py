import asyncio, json, httpx, grpc, logging
import chat_pb2, chat_pb2_grpc

TRT_URL = "http://localhost:6060/v1/chat/completions"
CPU_URL = "http://localhost:7070/v1/chat/completions"      # e.g. vLLM

logging.basicConfig(level=logging.INFO)

class Chat(chat_pb2_grpc.ChatServicer):
    async def StreamChat(self, request, context):
        url = TRT_URL if request.use_trt else CPU_URL
        payload = {
            "model":  "ensemble",
            "stream": True,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield chat_pb2.ChatDelta(text=delta)

async def serve():
    server = grpc.aio.server()
    chat_pb2_grpc.add_ChatServicer_to_server(Chat(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    logging.info("gRPC proxy on :50051")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
