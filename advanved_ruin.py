import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
from scipy import stats

results = np.array(["success", "failure"])
even_prob_arr = np.array([18/38, 20/38])
odd_prob_arr = np.array([18/38, 20/38])

def even_bet(money, bet):
    '''
    money : int : current bank value
    bet   : int : size of wager
    '''
    roll = random.choices(results, weights=even_prob_arr)[0]
    if money <= 0:
        return money
    if roll == "success":
        money += bet
    else:
        money -= bet
    return money

def flat_scheme_check(starting_money, base_bet):
    '''
    starting money : int : initial bank value
    base_bet       : int : amount for each bet
    ...
    net_tracker    : np.array : results of a single trial
    '''
    # Track net money at each spin
    net_tracker = np.array([])

    money = starting_money
    count = 0
    while (money > 0):
        money = even_bet(money, base_bet)
        count += 1
        net_tracker = np.append(net_tracker, money)
    return net_tracker

def martingale_scheme_check(starting_money, base_bet):
    '''
    starting money : int : initial bank value
    base_bet       : int : amount for each bet
    ...
    net_tracker    : np.array : results of a single trial
    '''
    # Track net money at each spin
    net_tracker = np.array([])

    money = starting_money
    current_bet = base_bet
    count = 0
    while (money > 0):
        previous_money = money
        money = even_bet(money, current_bet)
        count += 1
        if (money < previous_money):
            current_bet *= 2
        else:
            current_bet = base_bet

        net_tracker = np.append(net_tracker, money)
    return net_tracker

def fibonacci_scheme_check(starting_money, base_bet):
    '''
    starting money : int : initial bank value
    base_bet       : int : amount for each bet
    ...
    net_tracker    : np.array : results of a single trial
    '''

    # Track net money at each spin
    net_tracker = np.array([])

    money = starting_money
    fib_one = 0
    fib_two = 1
    count = 0
    while (money > 0):
        previous_money = money
        money = even_bet(money, (base_bet*fib_two))
        count += 1
        if (money < previous_money):
            saved_fib_one = fib_one
            fib_one = fib_two
            fib_two += saved_fib_one
        else:
            fib_two -= fib_one
            fib_one -= fib_two
            if(fib_one <= 0):
                fib_one = 0
                fib_two = 1
            if(fib_two <= 0):
                fib_one = 0
                fib_two = 1

        net_tracker = np.append(net_tracker, money)
    return net_tracker

def dalembert_scheme_check(starting_money, base_bet):
    '''
    starting money : int : initial bank value
    base_bet       : int : amount for each bet
    ...
    net_tracker    : np.array : results of a single trial
    '''

    # Track net money at each spin
    net_tracker = np.array([])

    money = starting_money
    current_bet = base_bet
    count = 0
    while (money > 0):
        previous_money = money
        money = even_bet(money, current_bet)
        count += 1

        if (money < previous_money):
            current_bet += base_bet
        elif (current_bet != base_bet):
            current_bet -= base_bet

        net_tracker = np.append(net_tracker, money)

    return net_tracker

def calculate_statistics(spins_until_ruin_array, strategy_name):
    '''
    Calculate and return statistical measures for a betting strategy
    
    spins_until_ruin_array : np.array : array of number of spins until ruin for each trial
    strategy_name          : str : name of the betting strategy
    
    Returns a dictionary with statistical measures
    '''
    n = len(spins_until_ruin_array)
    
    # Expected value (mean number of spins until ruin)
    expected_spins = np.mean(spins_until_ruin_array)
    
    # Variance and standard deviation
    variance = np.var(spins_until_ruin_array, ddof=1)
    std_dev = np.std(spins_until_ruin_array, ddof=1)
    
    # Median (50th percentile)
    median_spins = np.median(spins_until_ruin_array)
    
    # Quartiles
    q25 = np.percentile(spins_until_ruin_array, 25)
    q75 = np.percentile(spins_until_ruin_array, 75)
    
    # Min and Max
    min_spins = np.min(spins_until_ruin_array)
    max_spins = np.max(spins_until_ruin_array)
    
    # 95% Confidence Interval for the mean
    confidence_level = 0.95
    degrees_freedom = n - 1
    sample_mean = expected_spins
    sample_standard_error = std_dev / np.sqrt(n)
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, 
                                          sample_mean, sample_standard_error)
    
    # Coefficient of variation (relative variability)
    cv = (std_dev / expected_spins) * 100 if expected_spins > 0 else 0
    
    return {
        'strategy': strategy_name,
        'n_trials': n,
        'E[X]': expected_spins,
        'variance': variance,
        'std_dev': std_dev,
        'median': median_spins,
        'Q25': q25,
        'Q75': q75,
        'min': min_spins,
        'max': max_spins,
        'CI_lower': confidence_interval[0],
        'CI_upper': confidence_interval[1],
        'CV%': cv
    }

def print_statistics_table(stats_list):
    '''
    Print a formatted table of statistics for all strategies
    '''
    print("\n" + "="*100)
    print("GAMBLER'S RUIN STATISTICS - Expected Spins Until Ruin E[X]")
    print("="*100)
    print(f"{'Strategy':<15} {'Trials':<8} {'E[X]':<10} {'Std Dev':<10} {'Median':<10} {'Min':<8} {'Max':<8}")
    print("-"*100)
    
    for stat in stats_list:
        print(f"{stat['strategy']:<15} {stat['n_trials']:<8} {stat['E[X]']:<10.2f} "
              f"{stat['std_dev']:<10.2f} {stat['median']:<10.0f} "
              f"{stat['min']:<8.0f} {stat['max']:<8.0f}")
    
    print("="*100)
    print("\nDETAILED STATISTICS:")
    print("="*100)
    
    for stat in stats_list:
        print(f"\n{stat['strategy'].upper()}:")
        print(f"  Number of Trials: {stat['n_trials']}")
        print(f"  Expected Value E[X]: {stat['E[X]']:.2f} spins")
        print(f"  95% Confidence Interval: [{stat['CI_lower']:.2f}, {stat['CI_upper']:.2f}]")
        print(f"  Variance: {stat['variance']:.2f}")
        print(f"  Standard Deviation: {stat['std_dev']:.2f}")
        print(f"  Coefficient of Variation: {stat['CV%']:.2f}%")
        print(f"  Median: {stat['median']:.0f} spins")
        print(f"  25th Percentile: {stat['Q25']:.0f} spins")
        print(f"  75th Percentile: {stat['Q75']:.0f} spins")
        print(f"  Range: [{stat['min']:.0f}, {stat['max']:.0f}]")

def main(args):
    # Parse command line arguments
    num_trials = 1000  # Default number of trials
    if len(args) > 1:
        try:
            num_trials = int(args[1])
        except ValueError:
            print(f"Invalid number of trials: {args[1]}. Using default: {num_trials}")
    
    print(f"\nRunning {num_trials} trials for each betting strategy...")
    print("This may take a moment...\n")
    
    # Arrays to store spins until ruin for each trial
    flat_spins_list = []
    martingale_spins_list = []
    fibonacci_spins_list = []
    dalembert_spins_list = []
    
    # Run multiple trials
    for i in range(num_trials):
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{num_trials} trials...")
        
        # Run each strategy and record number of spins until ruin
        flat_counts = flat_scheme_check(100, 1)
        martingale_counts = martingale_scheme_check(100, 1)
        fibonacci_counts = fibonacci_scheme_check(100, 1)
        dalembert_counts = dalembert_scheme_check(100, 1)
        
        flat_spins_list.append(len(flat_counts))
        martingale_spins_list.append(len(martingale_counts))
        fibonacci_spins_list.append(len(fibonacci_counts))
        dalembert_spins_list.append(len(dalembert_counts))
    
    # Convert to numpy arrays
    flat_spins = np.array(flat_spins_list)
    martingale_spins = np.array(martingale_spins_list)
    fibonacci_spins = np.array(fibonacci_spins_list)
    dalembert_spins = np.array(dalembert_spins_list)
    
    # Calculate statistics for each strategy
    flat_stats = calculate_statistics(flat_spins, "Flat Bet")
    martingale_stats = calculate_statistics(martingale_spins, "Martingale")
    fibonacci_stats = calculate_statistics(fibonacci_spins, "Fibonacci")
    dalembert_stats = calculate_statistics(dalembert_spins, "D'Alembert")
    
    stats_list = [flat_stats, martingale_stats, fibonacci_stats, dalembert_stats]
    
    # Print statistics
    print_statistics_table(stats_list)
    
    # Create visualization of one representative trial
    print("\n\nGenerating visualization of a single representative trial...")
    
    # Use trials close to the median for visualization
    flat_trial_idx = np.argmin(np.abs(flat_spins - flat_stats['median']))
    martingale_trial_idx = np.argmin(np.abs(martingale_spins - martingale_stats['median']))
    fibonacci_trial_idx = np.argmin(np.abs(fibonacci_spins - fibonacci_stats['median']))
    dalembert_trial_idx = np.argmin(np.abs(dalembert_spins - dalembert_stats['median']))
    
    # Re-run those specific trials by using the same seed (or just run new ones)
    flat_counts = flat_scheme_check(100, 1)
    martingale_counts = martingale_scheme_check(100, 1)
    fibonacci_counts = fibonacci_scheme_check(100, 1)
    dalembert_counts = dalembert_scheme_check(100, 1)
    
    # Get largest number of rows  
    row_dim = max(
        len(flat_counts),
        len(martingale_counts),
        len(fibonacci_counts),
        len(dalembert_counts)
    )
    
    # Get a list of 1 ... row_dim, use this as a column in the data frame and the X-axis
    row_indices = list(range(1, row_dim + 1))
    index_vector = np.array(row_indices)
    
    df = pd.DataFrame([])
    
    # Spins column is just 1 ... max_spins
    df['Spins'] = index_vector
    
    # Include each trial, shorter ones are padded out by nan
    df['Flat'] = pd.Series(flat_counts).reindex(df.index)
    df['Martingale'] = pd.Series(martingale_counts).reindex(df.index)
    df['Fibonacci'] = pd.Series(fibonacci_counts).reindex(df.index)
    df['Dalembert'] = pd.Series(dalembert_counts).reindex(df.index)
    
    df.columns = ['Spin Number', 'Flat', 'Martingale', 'Fibonacci', 'Dalembert']
    
    # Create two subplots: one for bank over time, one for distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bank value over spins (single trial)
    ax1.plot(df['Spin Number'], df['Flat'], label='Flat Bet', linewidth=2, alpha=0.8)
    ax1.plot(df['Spin Number'], df['Martingale'], label='Martingale Bet', linewidth=2, alpha=0.8)
    ax1.plot(df['Spin Number'], df['Fibonacci'], label='Fibonacci Bet', linewidth=2, alpha=0.8)
    ax1.plot(df['Spin Number'], df['Dalembert'], label="D'Alembert Bet", linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Number of Spins', fontsize=12)
    ax1.set_ylabel('Bank', fontsize=12)
    ax1.set_title('Roulette Gambler\'s Ruin - Single Trial Example', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_ylim(-10, 200)
    
    # Plot 2: Distribution of spins until ruin
    positions = [1, 2, 3, 4]
    box_data = [flat_spins, martingale_spins, fibonacci_spins, dalembert_spins]
    labels = ['Flat', 'Martingale', 'Fibonacci', "D'Alembert"]
    
    bp = ax2.boxplot(box_data, positions=positions, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for mean in bp['means']:
        mean.set_color('purple')
        mean.set_linewidth(2)
    
    ax2.set_xlabel('Betting Strategy', fontsize=12)
    ax2.set_ylabel('Spins Until Ruin', fontsize=12)
    ax2.set_title(f'Distribution of Spins Until Ruin (n={num_trials} trials)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Add E[X] values as text
    for i, stat in enumerate(stats_list):
        ax2.text(i+1, ax2.get_ylim()[1]*0.95, f"E[X]={stat['E[X]']:.0f}", 
                ha='center', va='top', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("./outputs/gamblers-ruin-enhanced.png", dpi=150, bbox_inches='tight')
    print("Visualization saved to: gamblers-ruin-enhanced.png")
    
    # Create histogram comparison
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies_data = [
        (flat_spins, flat_stats, 'Flat Bet', colors[0]),
        (martingale_spins, martingale_stats, 'Martingale', colors[1]),
        (fibonacci_spins, fibonacci_stats, 'Fibonacci', colors[2]),
        (dalembert_spins, dalembert_stats, "D'Alembert", colors[3])
    ]
    
    for idx, (spins_data, stats, name, color) in enumerate(strategies_data):
        ax = axes[idx // 2, idx % 2]
        
        # Histogram
        n, bins, patches = ax.hist(spins_data, bins=50, alpha=0.7, color=color, edgecolor='black')
        
        # Add vertical lines for mean and median
        ax.axvline(stats['E[X]'], color='red', linestyle='--', linewidth=2, label=f"Mean (E[X])={stats['E[X]']:.0f}")
        ax.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median={stats['median']:.0f}")
        
        ax.set_xlabel('Spins Until Ruin', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} - Distribution of Spins Until Ruin', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("./outputs/gamblers-ruin-histograms.png", dpi=150, bbox_inches='tight')
    print("Histogram comparison saved to: gamblers-ruin-histograms.png")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv("./outputs/gamblers-ruin-statistics.csv", index=False)
    print("Statistics saved to: gamblers-ruin-statistics.csv")
    
    print("\n" + "="*100)
    print("Analysis complete!")
    print("="*100)

# Call main()
if __name__ == "__main__":
    main(sys.argv)
