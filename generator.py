# 🔧 Step 1: Import Required Library
import openai

# 🔐 Step 2: Set your OpenAI API key
openai.api_key = "api_key"
# 😊 Step 3: Hashtag Generator Function
def generate_hashtags(mood):
    hashtags = {
        'happy': '#joy #smile #happiness #positivevibes',
        'sad': '#feelingdown #mood #blues',
        'angry': '#anger #frustration #rage',
        'neutral': '#calm #neutral #peaceful',
        'excited': '#thrilled #pumped #cantwait',
        'relaxed': '#chillvibes #relaxation #serenity'
    }
    return hashtags.get(mood.lower(), '#mood')

# 🧠 Step 4: Caption Generator using GPT-3 (or GPT-4 if available)
def generate_caption(mood):
    prompt = f"Write a creative and relatable Instagram caption for someone feeling {mood}."
    response = openai.Completion.create(
        engine="text-davinci-003",  # or "gpt-4" if you're using GPT-4
        prompt=prompt,
        max_tokens=50,
        temperature=0.8,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

# 🧪 Step 5: Input from user
user_mood = input("Enter your mood (happy, sad, angry, etc.): ")

# 🖼️ Step 6: Output both caption and hashtags
caption = generate_caption(user_mood)
hashtags = generate_hashtags(user_mood)

print("\n🎯 Suggested Caption:")
print(caption)

print("\n🏷️ Suggested Hashtags:")
print(hashtags)
