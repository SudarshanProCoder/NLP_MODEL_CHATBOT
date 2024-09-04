import json

# Sample dataset with intents and examples
training_data = {
    "intents": [
        {
            "tag": "book_ticket",
            "patterns": ["I want to book a ticket", "Book a ticket to Paris", "Can you book a ticket for me?"]
        },
        {
            "tag": "cancel_ticket",
            "patterns": ["I want to cancel my ticket", "Cancel my reservation", "Can you cancel my ticket?"]
        },
        {
            "tag": "check_availability",
            "patterns": ["Is there any ticket available?", "Check availability for New York", "Are there seats available?"]
        }
    ]
}

# Save the data as a JSON file
with open('intents.json', 'w') as f:
    json.dump(training_data, f, indent=4)
