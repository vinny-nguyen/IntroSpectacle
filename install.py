import cohere

co = cohere.Client("qNAfz59ZFGtOxOwEcm66pDZj1XiBNp2CwAbxiFi8")

response = co.chat(
    model="command-r-plus",
	message="Write me a short summary for this person with several bullet points for interesting points about them."
)

print(response.text)
