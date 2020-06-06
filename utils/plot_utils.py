import matplotlib.pyplot as plt
import seaborn as sns

def countplots_with_multiple_categories(df, target: str, text: str):
    """
    Plots multiple countplots with shared context.
    Meaningful only when num of categorical variables <= 6 and >= 2

    df: dataframe
    hue: target
    text: shared text among columns.
    """

    cols = [col for col in df.columns if text in col]

    if len(cols) <= 3:
        figsize = (24, 7)
        ncols = len(cols)
        nrows = 1
    elif len(cols) > 3:
        figsize = (24, 24)
        ncols = 2
        nrows = int(len(cols) / 2) + (len(cols) % 2 > 0)
    else:
        print("Categorical variables are out of defined size")

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.countplot(x=col, hue=target, data=df, ax=ax)
    plt.tight_layout()
    plt.show()


def boxplots_with_multiple_categories(df, target: str, text: str):
    """
    Plots multiple boxplots with shared context.
    Meaningful only when num of categorical variables <= 6 and >= 2

    df: dataframe
    x: col
    y: target
    text: shared text among columns.
    """

    cols = [col for col in df.columns if text in col]

    if len(cols) <= 3:
        figsize = (24, 7)
        ncols = len(cols)
        nrows = 1
    elif len(cols) > 3:
        figsize = (24, 24)
        ncols = 2
        nrows = int(len(cols) / 2) + (len(cols) % 2 > 0)
    else:
        print("Categorical variables are out of defined size")

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharey=True)
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.boxplot(x=col, y=target, data=df, ax=ax)
    plt.tight_layout()
    plt.show()