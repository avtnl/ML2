fig, axs = plt.subplots(1,2, figsize=(12, 4))
axs = axs.ravel()

# c1=['red' if x[0] == 'Peruzzi' else 'grey' for x in deg]
# c2=['red' if x[0] == 'Peruzzi' else 'grey' for x in clust]
c1=['red' if (x[0] == 'Strozzi' or x[0] == 'Medici') else 'grey' for x in deg]
c2=['red' if (x[0] == 'Strozzi' or x[0] == 'Medici') else 'grey' for x in clust]

axs[0].bar([x[0] for x in deg], [x[1] for x in deg], color=c1);
axs[0].set_title('Degree')

axs[1].bar([x[0] for x in clust], [x[1] for x in clust], color=c2);
axs[1].set_title('Clustering coefficient')

for ax in axs:
    ax.tick_params(axis='x', rotation=45)
plt.suptitle("The Medici family has a low clustering coefficient")
plt.tight_layout()