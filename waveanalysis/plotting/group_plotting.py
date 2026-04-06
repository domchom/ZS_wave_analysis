import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def generate_group_comparison(
    summary_df: pd.DataFrame,
    log_params: dict,
    dark_plots: bool = False
) -> dict:
    """
    Generate group comparison plots for each parameter in the summary dataframe.

    Parameters:
        summary_df (pd.DataFrame): The summary dataframe containing the data for comparison.
                                   Must contain a column 'Group Name' and columns with 'Mean' in their names.
        log_params (dict): A dictionary to log any errors encountered during plotting.
                           Expects a key 'Plotting errors' with a list as value.
        dark_plots (bool): If True, use a dark theme with black background.

    Returns:
        dict: A dictionary mapping parameter name -> matplotlib Figure.
    """
    print('Generating group comparisons...')
    group_mean_parameter_figs = {}

    # get the parameters to compare
    parameters_to_compare = [column for column in summary_df.columns if 'Mean' in column]

    # choose style consistent with your other functions
    style = 'dark_background' if dark_plots else 'default'

    for param in parameters_to_compare:
        try:
            # skip if column is completely NaN or empty after dropna
            if summary_df[param].dropna().empty:
                raise ValueError("No data to compare")

            with plt.style.context(style):
                fig, ax = plt.subplots()

                # Force background to black in dark mode
                if dark_plots:
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')

                # boxplot
                sns.boxplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    showfliers=False,
                    ax=ax
                )

                # swarmplot (jittered points)
                sns.swarmplot(
                    x='Group Name',
                    y=param,
                    data=summary_df,
                    color=".25",
                    ax=ax
                )

                ax.set_title(param)
                ax.set_xlabel('Group')
                ax.set_ylabel(param)

                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                fig.tight_layout()
                group_mean_parameter_figs[param] = fig
                plt.close(fig)

        except ValueError:
            # make sure the key exists
            if 'Plotting errors' not in log_params:
                log_params['Plotting errors'] = []
            log_params['Plotting errors'].append(f'No data to compare for {param}')

    return group_mean_parameter_figs
