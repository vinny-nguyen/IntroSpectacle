import cohere

co = cohere.Client(api_key="<YOUR API KEY>")

response = co.chat(
    model="command-r-plus",
	message="Write me a short summary for this person with several bullet points for interesting points about them."
)

print(response.text)
