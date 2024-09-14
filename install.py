import cohere

co = cohere.Client('COHERE_API_KEY')

response = co.chat(
    model="command-r-plus",
	message="Write me a short summary for this person with several bullet points for interesting points about them.",
    conversation_id='user_id',
    document = """"""
)

print(response.text)
