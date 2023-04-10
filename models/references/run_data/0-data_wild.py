from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
'''
dataset_names_wilds = ['globalwheat', 'rxrx1', 'poverty', 'amazon', 'camelyon17', 'civilcomments', 'iwildcam',  'ogb-molpcba', 'fmow', 'py150', 'celebA', 'domainnet', 'waterbirds', 'yelp', 
'bdd100k', 'sqf', 'encode']
'''
dataset_names = [ 'rxrx1', 'poverty', 'camelyon17', 'iwildcam']
for name in dataset_names:
    dataset = get_dataset(dataset=name, download=True)

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
'''
mv ~/.vscode-server /gpfs/scratch/choy07/.vscode-server
ln -s /gpfs/scratch/choy07/.vscode-server ~/.vscode-server
'''
# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)

# (Optional) Load unlabeled data
dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
unlabeled_data = dataset.get_subset(
    "test_unlabeled",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)
unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# Train loop
for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
    x, y, metadata = labeled_batch
    unlabeled_x, unlabeled_metadata = unlabeled_batch
    ...