import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Input, LeakyReLU, Reshape
from tensorflow.keras.models import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def to_scalar(val):
    """Safely extracts a scalar float from any 0D/1D numpy array, list, or tensor."""
    arr = np.asarray(val)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.flatten()[0])


def load_hazard_signs(filename="q29_hazard_signs_cgan.csv"):
    df = pd.read_csv(get_path(filename))
    X = df[[f"p_{i}" for i in range(64)]].values.astype(np.float32)
    y = df["Sign_Class"].values
    return X, y


def build_cgan_models(latent_dim=12, num_classes=3):
    # Generator
    noise_in = Input(shape=(latent_dim,))
    label_in = Input(shape=(1,), dtype="int32")
    label_emb = Reshape((latent_dim,))(Embedding(num_classes, latent_dim)(label_in))
    gen_in = Concatenate()([noise_in, label_emb])
    h_g = Dense(32)(gen_in)
    h_g = LeakyReLU(alpha=0.2)(h_g)
    out_g = Dense(64, activation="sigmoid")(h_g)
    generator = Model([noise_in, label_in], out_g, name="cgen")

    # Discriminator
    img_in = Input(shape=(64,))
    disc_label_in = Input(shape=(1,), dtype="int32")
    disc_label_emb = Reshape((64,))(Embedding(num_classes, 64)(disc_label_in))
    disc_in = Concatenate()([img_in, disc_label_emb])
    h_d = Dense(32)(disc_in)
    h_d = LeakyReLU(alpha=0.2)(h_d)
    out_d = Dense(1, activation="sigmoid")(h_d)
    discriminator = Model([img_in, disc_label_in], out_d, name="cdisc")
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy"
    )

    # Combined Conditional Model (trains generator)
    discriminator.trainable = False
    validity = discriminator([out_g, label_in])
    cgan = Model([noise_in, label_in], validity)
    cgan.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy"
    )

    return generator, discriminator, cgan


def train_cgan(X_real, y_real, epochs=45, batch_size=32, latent_dim=12):
    gen, disc, cgan = build_cgan_models(latent_dim)
    m = len(X_real)
    d_losses, g_losses = [], []

    for _ in range(epochs):
        # Sample real images and labels
        idx = np.random.randint(0, m, batch_size)
        real_imgs, labels = X_real[idx], y_real[idx].reshape(-1, 1)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_imgs = gen.predict([noise, labels], verbose=0)

        # Train Discriminator
        res_real = disc.train_on_batch([real_imgs, labels], np.ones((batch_size, 1)) * 0.9)
        res_fake = disc.train_on_batch([fake_imgs, labels], np.zeros((batch_size, 1)))

        d_loss_real = to_scalar(res_real)
        d_loss_fake = to_scalar(res_fake)
        d_losses.append(0.5 * (d_loss_real + d_loss_fake))

        # Train Generator
        sampled_labels = np.random.randint(0, 3, (batch_size, 1))
        g_res = cgan.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))
        g_losses.append(to_scalar(g_res))

    return gen, d_losses, g_losses


def generate_outputs(gen, d_losses, g_losses, latent_dim=12):
    class_names = ["Speed_Limit (Ring)", "Caution (Triangle)", "Stop (Block)"]

    # Table 1: Loss Progression
    loss_df = pd.DataFrame(
        {
            "Epoch": range(1, len(d_losses) + 1),
            "D_Loss": np.round(d_losses, 4),
            "G_Loss": np.round(g_losses, 4),
        }
    )
    loss_df.iloc[::5].to_csv(get_path("q29_table1_cgan_loss.csv"), index=False)

    # Plot 1: Loss Progression Curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d_losses, label="Conditional Discriminator Loss", color="royalblue")
    ax.plot(g_losses, label="Conditional Generator Loss", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("cGAN Loss Dynamics for Class-Conditioned Synthesis")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_path("q29_plot1_cgan_loss.png"))
    plt.close()

    # Plot 2: Class-Conditioned Generated Sign Samples
    fig2, axes = plt.subplots(3, 4, figsize=(8, 6))
    for c in range(3):
        noise = np.random.normal(0, 1, (4, latent_dim))
        c_inputs = np.full((4, 1), c)
        syn_signs = gen.predict([noise, c_inputs], verbose=0)
        for col in range(4):
            ax_img = axes[c, col]
            ax_img.imshow(syn_signs[col].reshape(8, 8), cmap="magma", vmin=0, vmax=1)
            ax_img.axis("off")
            if col == 0:
                ax_img.set_title(class_names[c], loc="left", fontsize=9, fontweight="bold")
    plt.suptitle("cGAN Synthesized Signs Conditioned on Category")
    plt.tight_layout()
    plt.savefig(get_path("q29_plot2_class_samples.png"))
    plt.close()


if __name__ == "__main__":
    X_signs, y_signs = load_hazard_signs()
    generator, d_loss_hist, g_loss_hist = train_cgan(X_signs, y_signs)
    generate_outputs(generator, d_loss_hist, g_loss_hist)
    print("Lab 29 complete: 1 table and 2 plots saved in Exp.No.29.")