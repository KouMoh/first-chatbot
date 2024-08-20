from langchain_intro.chatbot import review_chain

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

result = review_chain.invoke({"context": context, "question": question})
print(result)