import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge
import warnings
import random

class CNNDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes):
        super(CNNDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, sequence_length, embedding_dim]
        x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, sequence_length, embedding_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply each convolution
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Max-over-time pooling
        x = torch.cat(x, 1)  # Concatenate the features
        x = self.dropout(x)  # Apply dropout
        logits = self.fc(x)  # Fully connected layer to get logits
        return torch.sigmoid(logits)

# Filter deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the dataset and create a subset
dataset = load_dataset("cnn_dailymail", "3.0.0")
shuffled_dataset = dataset['train'].shuffle(seed=42)
subset_size = 100  # Define subset size
train_subset = shuffled_dataset.select(range(subset_size))

# Initialize models and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
rouge = Rouge()

def train_gan(generator, discriminator, tokenizer, train_data, device, epochs=1):
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, sample in enumerate(train_data):
            # Prepare summaries
            real_summary = tokenizer(sample['highlights'], return_tensors="pt", padding='max_length', max_length=100, truncation=True)
            real_summary = real_summary['input_ids'].to(device)
            inputs = tokenizer(sample['article'], return_tensors="pt", max_length=1024, truncation=True)
            fake_summary_ids = generator.generate(inputs['input_ids'].to(device), num_beams=4, max_length=100, early_stopping=True)
            fake_summary = tokenizer.decode(fake_summary_ids[0], skip_special_tokens=True)
            fake_summary_ids = tokenizer(fake_summary, return_tensors="pt", padding='max_length', max_length=100, truncation=True)
            fake_summary_ids = fake_summary_ids['input_ids'].to(device)

            # Discriminator training
            disc_optimizer.zero_grad()
            real_output = discriminator(real_summary)
            real_loss = criterion(real_output, torch.ones_like(real_output))
            real_loss.backward()
            fake_output = discriminator(fake_summary_ids)
            fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
            fake_loss.backward()
            disc_optimizer.step()

            # Generator training
            gen_optimizer.zero_grad()
            fake_output = discriminator(fake_summary_ids)
            gen_loss = criterion(fake_output, torch.ones_like(fake_output))
            gen_loss.backward()
            gen_optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Sample {i}, Gen Loss: {gen_loss.item()}, Disc Real Loss: {real_loss.item()}, Disc Fake Loss: {fake_loss.item()}")
                # Compute ROUGE scores
                scores = rouge.get_scores(fake_summary, sample['highlights'])
                print("ROUGE-1:", scores[0]['rouge-1']['f'])
                print("ROUGE-2:", scores[0]['rouge-2']['f'])
                print("ROUGE-L:", scores[0]['rouge-l']['f'])

# Initialize the discriminator
discriminator = CNNDiscriminator(tokenizer.vocab_size, 256, 100, [3, 4, 5]).to(device)

# Train on subset
train_gan(generator, discriminator, tokenizer, train_subset, device)
