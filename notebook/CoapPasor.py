import logging
import asyncio


from aiocoap import *

logging.basicConfig(level=logging.INFO)

async def main():
    protocol = await Context.create_client_context()
    requests = Message(code=GET, uri='coap://localhost:9988')

    try:
        response = await protocol.request(requests).response
    except Exception as e:
        print(e)
    else:
        print('Results: %s\n%r%', (response.code, response.payload))
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
