import pandas as pd
import matplotlib.pyplot as plt


def main():
    try:
        df_data = pd.read_csv('moons.csv')
        df_grid = pd.read_csv('moons_decision_boundary.csv')
    except FileNotFoundError as e:
        print(f"could not find file\n{e}")
        return

    X = df_data[['x', 'y']].values
    y = df_data['label'].values
    y_norm = (y > 0).astype(int)

    grid_x = df_grid['x'].values
    grid_y = df_grid['y'].values
    grid_z = (df_grid['score'].values > 0).astype(int)

    x_min, x_max = grid_x.min(), grid_x.max()
    y_min, y_max = grid_y.min(), grid_y.max()

    # plot1
    plt.figure(figsize=(8, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y_norm, s=40, cmap=plt.cm.Spectral, edgecolors='k')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Moons Dataset (Raw)')

    plt.savefig('moons_dataset.png', dpi=300)
    plt.close()

    # plot 2
    plt.figure(figsize=(8, 6))

    plt.tricontourf(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y_norm, s=40, cmap=plt.cm.Spectral, edgecolors='k')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Decision Boundary')

    plt.savefig('decision_boundary.png', dpi=300)
    plt.close()

    print("saved plots as images")


if __name__ == '__main__':
    main()
