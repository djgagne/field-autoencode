import numpy as np
import matplotlib.pyplot as plt
from fieldae.fields import random_field_generator
from fieldae.models import VariationalAutoEncoder

def main():
    test_VariationalAutoEncoder()

def test_VariationalAutoEncoder():
    np.random.seed(253)
    x_grid, y_grid = np.meshgrid(np.arange(16), np.arange(16))
    rfg = random_field_generator(x_grid, y_grid, length_scales=[16.0])
    x = np.zeros((4096, 16, 16, 1), dtype=np.float32)
    for i in range(x.shape[0]):
        x[i] = next(rfg)
    x_scaled = (x - x.min()) / (x.max() - x.min())
    vae = VariationalAutoEncoder(latent_dim=8, min_filters=16, learning_rate=0.001, hidden_activation="leaky",
                                 verbose=2)
    vae.fit(x_scaled, epochs=10, batch_size=256)
    z_mean, z_var = vae.encode(x_scaled[:128])
    sample = vae.reparameterize(z_mean, z_var)
    x_new = vae.decode(sample, apply_sigmoid=True).numpy()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x_scaled[25, :, :, 0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    print(x_new[25].min(), x_new[25].max())
    print(x_scaled[25].min(), x_scaled[25].max())
    plt.pcolormesh(x_new[25, :, :, 0])
    plt.colorbar()
    plt.savefig("test.png", dpi=200)
    return

if __name__ == "__main__":
    main()
