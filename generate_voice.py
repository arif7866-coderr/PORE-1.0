import edge_tts
import asyncio

async def generate_voice():
    text = "Welcome to PORE."

    # 🎤 Natural, clear, professional male voice
    tts = edge_tts.Communicate(
        text,
        voice="en-US-ChristopherNeural",  # male, professional tone
        rate="-20%",   # slightly slower for emphasis
        pitch="+1Hz"   # a touch higher for clarity
    )

    await tts.save("welcome_pore.mp3")

asyncio.run(generate_voice())
