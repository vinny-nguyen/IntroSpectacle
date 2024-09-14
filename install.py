import cohere

co = cohere.Client("qNAfz59ZFGtOxOwEcm66pDZj1XiBNp2CwAbxiFi8")

response = co.chat(
	message="hello world!"
)

print(response)
