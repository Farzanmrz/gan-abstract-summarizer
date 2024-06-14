# test.py
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from discriminator import Discriminator
import evaluate
import os

# Load ROUGE metric
rouge = evaluate.load('rouge')

# Load the CNN/Daily Mail dataset and select 20 samples
dataset = load_dataset("cnn_dailymail", "3.0.0", split = "train[:20]")

# Split the dataset into train (80%), validation (10%), and test (10%)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [ train_size, val_size, test_size ])

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Initialize the Discriminator
vocab_size = tokenizer.vocab_size
embed_size = 768  # Embedding size for BART
discriminator = Discriminator(vocab_size, embed_size).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

# Training parameters
loss_fn = torch.nn.CrossEntropyLoss()
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = 1e-4)
optimizer_g = torch.optim.Adam(generator.parameters(), lr = 5e-5)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Load model parameters if they exist
if os.path.exists("generator_checkpoint.pth"):
	generator.load_state_dict(torch.load("generator_checkpoint.pth", map_location = device))
if os.path.exists("discriminator_checkpoint.pth"):
	discriminator.load_state_dict(torch.load("discriminator_checkpoint.pth", map_location = device))

# Training loop
num_epochs = 3
batch_size = 2
beta = 0.5  # Balance factor between policy gradient loss and max-likelihood loss


# Save the original summary once
def save_original_summary():
	first_article = test_dataset[ 0 ][ 'article' ]
	original_summary = test_dataset[ 0 ][ 'highlights' ]
	with open("first_article_summaries.txt", "w") as f:
		f.write(f"Original Summary:\n{original_summary}\n\n")


# Ensure original summary is saved once
if not os.path.exists("first_article_summaries.txt"):
	save_original_summary()


def train_discriminator():
	discriminator.train()
	total_loss = 0
	for batch in DataLoader(train_dataset, batch_size = batch_size, shuffle = True):
		# Generate machine summaries
		inputs = tokenizer(batch[ 'article' ], return_tensors = "pt", padding = True, truncation = True, max_length = 1024).to(device)
		outputs = generator.generate(inputs[ 'input_ids' ], max_length = 150, min_length = 40, length_penalty = 2.0, num_beams = 4, early_stopping = True)
		machine_summaries = [ tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False) for g in outputs ]

		# Prepare discriminator inputs
		human_summaries = batch[ 'highlights' ]
		summaries = human_summaries + machine_summaries
		labels = torch.tensor([ 1 ] * len(human_summaries) + [ 0 ] * len(machine_summaries)).to(device)

		encoding = tokenizer(summaries, return_tensors = 'pt', padding = True, truncation = True, max_length = 1024).to(device)
		input_ids = encoding[ 'input_ids' ]

		# Train discriminator
		optimizer_d.zero_grad()
		predictions = discriminator(input_ids)
		loss = loss_fn(predictions, labels)
		loss.backward()
		optimizer_d.step()

		total_loss += loss.item()
	return total_loss / len(train_dataset)


def train_generator():
	generator.train()
	total_loss = 0
	for batch in DataLoader(train_dataset, batch_size = batch_size, shuffle = True):
		# Generate machine summaries
		inputs = tokenizer(batch[ 'article' ], return_tensors = "pt", padding = True, truncation = True, max_length = 1024).to(device)
		optimizer_g.zero_grad()

		outputs = generator.generate(inputs[ 'input_ids' ], max_length = 150, min_length = 40, length_penalty = 2.0, num_beams = 4, early_stopping = True)
		machine_summaries = [ tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False) for g in outputs ]

		# Prepare discriminator inputs
		encoding = tokenizer(machine_summaries, return_tensors = 'pt', padding = True, truncation = True, max_length = 1024).to(device)
		input_ids = encoding[ 'input_ids' ]

		# Calculate J_pg using the policy gradient theorem
		predictions = discriminator(input_ids)
		rewards = predictions[ :, 1 ]  # Use discriminator's probability of being real as the reward
		log_probs = torch.log(predictions[ :, 1 ] + 1e-10)  # Avoid log(0) issues

		j_pg = -rewards.mean() * log_probs.mean()  # Policy gradient loss (negative reward for better alignment)

		# Maximum likelihood loss (cross-entropy)
		labels = tokenizer(batch[ 'highlights' ], return_tensors = "pt", padding = True, truncation = True, max_length = 1024).to(device)[ 'input_ids' ]
		outputs = generator(inputs[ 'input_ids' ], labels = labels)
		ml_loss = outputs.loss

		# Total generator loss
		loss = beta * j_pg + (1 - beta) * ml_loss
		loss.backward()
		optimizer_g.step()

		total_loss += loss.item()
	return total_loss / len(train_dataset)


def evaluate_generator( epoch ):
	generator.eval()
	summaries = [ ]
	references = [ ]
	first_article_summary = None
	for i, batch in enumerate(DataLoader(test_dataset, batch_size = batch_size)):
		inputs = tokenizer(batch[ 'article' ], return_tensors = "pt", padding = True, truncation = True, max_length = 1024).to(device)
		outputs = generator.generate(inputs[ 'input_ids' ], max_length = 150, min_length = 40, length_penalty = 2.0, num_beams = 4, early_stopping = True)
		batch_summaries = [ tokenizer.decode(g, skip_special_tokens = True, clean_up_tokenization_spaces = False) for g in outputs ]
		summaries += batch_summaries
		references += batch[ 'highlights' ]

		# Store the summary of the first article
		if i == 0:
			first_article_summary = batch_summaries[ 0 ]

	# Compute ROUGE scores
	results = rouge.compute(predictions = summaries, references = references)

	# Save the summary of the first article
	with open("first_article_summaries.txt", "a") as f:
		f.write(f"Epoch {epoch + 1}:\n{first_article_summary}\n\n")

	return results


for epoch in range(num_epochs):
	print(f"Epoch {epoch + 1}/{num_epochs}")
	d_loss = train_discriminator()
	g_loss = train_generator()
	rouge_scores = evaluate_generator(epoch)
	print(f"Discriminator Loss: {d_loss}")
	print(f"Generator Loss: {g_loss}")
	print(f"ROUGE Scores: {rouge_scores}")

	# Save the model checkpoints
	torch.save(generator.state_dict(), "generator_checkpoint.pth")
	torch.save(discriminator.state_dict(), "discriminator_checkpoint.pth")
