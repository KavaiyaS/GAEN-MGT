import matplotlib.pyplot as plt
def save_bar(values, labels, title, path):
    plt.figure(figsize=(8,5)); plt.bar(labels, values); plt.title(title); plt.ylabel("Score")
    plt.tight_layout(); plt.savefig(path, dpi=200)
