
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from PIL import Image
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                nn.ConvTranspose2d(100, 256, kernel_size=4, stride=1, padding=0),
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2, inplace=True),

                                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2, inplace=True),
                                

                                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2, inplace=True),

                                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                
                                nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=1),
                                nn.Tanh()
                                )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2, inplace=True),

                                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2, inplace=True),
                                
                                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2, inplace=True),

                                nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
                                )
    def forward(self, x):
        return self.net(x).view(-1,1)

def plot_image(generator, num_images=5, const_noise=None, epoch=None):
    generator.eval()
    
    if const_noise is not None:
        with torch.no_grad():
            img = generator(const_noise).detach().squeeze().cpu()
        img = (img + 1) / 2
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img, mode='RGBA')
        pil_img.save(f'output/long/generated_image_{epoch}.png')

        # plt.figure(figsize=(6, 6))
        # plt.imshow(img_np)
        # plt.title("Generated Image")
        # plt.axis('off')
        # plt.savefig(f'generated_image_{epoch}.png')
        # plt.close()

    for i in range(num_images):
        noise = torch.randn(1, 100, 1, 1, device=device)
        with torch.no_grad():
            img = generator(noise).detach().squeeze().cpu()
        img = (img + 1) / 2
        img = (img * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(img, mode='RGBA')
        pil_img.save(f'output/long/generated_image_{i+1}.png')

def plot_loss(g_losses, d_losses_real, d_losses_fake):
    with open("output/long/g_losses.json", "w") as f:
        json.dump(g_losses, f)
    with open("output/long/d_losses_real.json", "w") as f:
        json.dump(d_losses_real, f)
    with open("output/long/d_losses_fake.json", "w") as f:
        json.dump(d_losses_fake, f)

    plt.figure(figsize=(8, 6))
    plt.plot(g_losses, label="Generator Loss", color='green')
    plt.plot(d_losses_real, label="Discriminator Real Loss", color='red')
    plt.plot(d_losses_fake, label="Discriminator Fake Loss", color='blue')
    # plt.ylim(0, max(g_losses[-1], d_losses[-1])*1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/long/gan_losses.png")
    plt.close()

def rgba_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')

if __name__ == "__main__":

    transform = transforms.Compose([
                                    transforms.ToTensor(), # converts [0,255] to [0, 1]
                                    transforms.Normalize([0.5]*4, [0.5]*4) # converts to [-1, 1] for using tanh
                                    ])

    dataset = datasets.ImageFolder(root='joypixels-7.0-free/png/labeled/32/', transform=transform, loader=rgba_loader)

    # generator = Generator().to(device)
    # discriminator = Discriminator().to(device)

    generator = torch.load('output/new/generator.pt', weights_only=False)
    discriminator = torch.load('output/new/discriminator.pt', weights_only=False)

    # generator.eval()
    # for i in range(10):

    #     with torch.no_grad():
    #         noise = torch.randn(1, 100, 1, 1, device=device)
    #         img = generator(noise).detach().cpu().squeeze()
    #         img = (img * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    #         import matplotlib.pyplot as plt
    #         plt.imshow(img)
    #         plt.title("Sample from Loaded Generator")
    #         plt.axis("off")
    #         plt.show()
    # exit()


    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # d_optimizer = torch.optim.RMSprop(discriminator.parameters(), 0.5e-3)
    # g_optimizer = torch.optim.RMSprop(generator.parameters(), 1e-4)

    epochs = 10000
    batch_size = 512

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    g_losses = []
    d_losses_real = []
    d_losses_fake = []

    const_noise = torch.randn(1, 100, 1, 1, device=device)
    print('Started...')
    for i in range(999, epochs):
        
        
        total_g_loss = 0
        total_d_loss_real = 0
        total_d_loss_fake = 0
        batches = 0
        for batch in tqdm(train_loader):
            real_imgs = batch[0].to(device)
            # real_labels = torch.ones(real_imgs.shape[0], 1, device=device)
            real_labels = real_labels = torch.full((real_imgs.shape[0], 1), fill_value=0.9, device=device)



            # img = real_imgs[0]
            # print(img.shape)
            # img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
            # pil_img = Image.fromarray(img, mode='RGBA')
            # pil_img.save(f'generated_image.png')
            # exit()

            discriminator.train()
            d_optimizer.zero_grad()
            d_pred = discriminator(real_imgs)
            d_loss_real = criterion(d_pred, real_labels)
            d_loss_real.backward()
            d_optimizer.step()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_imgs = generator(noise)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # X = torch.cat((real_imgs, fake_imgs), dim=0)
            # y = torch.cat((real_labels, fake_labels), dim=0)

            # indices = torch.randperm(X.size(0), device=device) # for shuffling
            # X = X[indices]
            # y = y[indices]

            discriminator.train()
            d_optimizer.zero_grad()
            d_pred = discriminator(fake_imgs)
            d_loss_fake = criterion(d_pred, fake_labels)
            d_loss_fake.backward()
            d_optimizer.step()

            generator.train()
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            g_pred = discriminator(generator(noise))
            g_loss = criterion(g_pred, torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss_real += d_loss_real.item()
            total_d_loss_fake += d_loss_fake.item()
            batches += 1

        avg_g_loss = total_g_loss / batches
        avg_d_loss_real = total_d_loss_real / batches
        avg_d_loss_fake = total_d_loss_fake / batches
        g_losses.append(avg_g_loss)
        d_losses_real.append(avg_d_loss_real)
        d_losses_fake.append(avg_d_loss_fake)

        plot_loss(g_losses, d_losses_real, d_losses_fake)
        print(f"[epoch {i+1}] Generator Loss: {avg_g_loss:.4f}, Discriminator Real Loss: {avg_d_loss_real:.4f}, Discriminator Fake Loss: {avg_d_loss_fake:.4f}")
        # torch.cuda.empty_cache()
        if (i+1) % 50 == 0:
            plot_image(generator, const_noise=const_noise, epoch=i+1)
            torch.save(generator, 'output/long/generator.pt')
            torch.save(discriminator, 'output/long/discriminator.pt')
            torch.cuda.empty_cache()

        
