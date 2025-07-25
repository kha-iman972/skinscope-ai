import os

total = 0
for root, _, files in os.walk('data'):
    total += sum(1 for f in files if f.lower().endswith(('.jpg', '.png')))
print("Total training images:", total)
