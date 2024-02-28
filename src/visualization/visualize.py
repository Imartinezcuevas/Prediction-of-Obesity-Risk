import pandas as pd
import matplotlib. pyplot as plt
import seaborn as sns

def plot_count(df: pd.core.frame.DataFrame, col: str, inner_colors: list=['#f6bb97', '#f4865e', '#ec4a3e', '#ca1a50', '#951c5b', '#601f52', '#2e1739'], title_name: str='Train') -> None:
    # Set background color
    
    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = df[col].value_counts()
    N = len(s1)

    inner_sizes = s1/N

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.1f%%',
        pctdistance=0.83, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='rocket_r', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

def bar_plot_all_categorical(df: pd.core.frame.DataFrame, categorical_cols: list, hue: str) -> None:
    plt.figure(figsize=(14, len(categorical_cols)*3))

    for i, col in enumerate(categorical_cols):
        
        plt.subplot(len(categorical_cols)//2 + len(categorical_cols) % 2, 2, i+1)
        ax = sns.countplot(x=col, hue=hue, data=df, palette='rocket_r')
        for container in ax.containers:
            ax.bar_label(container)
        plt.title(f"{col} countplot by ", fontweight = 'bold')
        plt.ylim(0, df[col].value_counts().max() + 10)
        
    plt.tight_layout()
    plt.show()