import os
from pybaseball import statcast_pitcher
from pybaseball.playerid_lookup import playerid_lookup
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def process_data_ff(data_dir):
    """
    指定されたディレクトリ内の全CSVファイルを読み込み、処理を行う関数
    
    Parameters
    ----------
    data_dir : str
        データが保存されているディレクトリパス
        
    Returns
    -------
    pandas.DataFrame
        処理済みの全選手データ
    """
    # スクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # データディレクトリのパスを作成
    data_path = os.path.join(script_dir, data_dir)
    
    # データディレクトリの存在確認
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} does not exist")

    # データ格納
    all_data = pd.DataFrame()

    # ディレクトリ内の全CSVファイルを処理
    for file_name in os.listdir(data_path):
        if not file_name.endswith('.csv'):
            continue
            
        file_path = os.path.join(data_path, file_name)
        print(f"Loading data from {file_name}")
        df = pd.read_csv(file_path)
        
        # データの加工
        df = df[df["pitch_type"] == "FF"]
        df = df[df["release_spin_rate"] > 1800]  # 回転数フィルタ
        df = df[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'release_pos_x', 'release_pos_z', 'arm_angle', 'spin_axis', 'vx0', 'vy0', 'vz0', 'p_throws']].dropna()

        # データが空の場合はスキップ
        if df.empty:
            print(f"No data after filtering for {file_name}")
            continue

        # 単位変換：mph → km/h, feet → m
        MPH_TO_KMH = 1.60934
        FEET_TO_M = 0.3048
        df['release_speed'] = df['release_speed'] * MPH_TO_KMH
        if df['p_throws'].iloc[0] == 'L':
            df['plate_x'] = -df['plate_x']
            df['release_pos_x'] = -df['release_pos_x']
            df['vx0'] = -df['vx0']
            df['vy0'] = -df['vy0']
            df['vz0'] = -df['vz0']
        df['plate_x'] = df['plate_x'] * FEET_TO_M
        df['plate_z'] = df['plate_z'] * FEET_TO_M
        df['release_pos_x'] = df['release_pos_x'] * FEET_TO_M
        df['release_pos_z'] = df['release_pos_z'] * FEET_TO_M
        df["player"] = file_name.split('_')[0]  # ファイル名から選手名を抽出
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    return all_data

def ellipse_plot(eigenvectors, eigenvalues, mean, color, linestyle='-'): # plot_ellipse
    """
    固有値・固有ベクトルから楕円を描画する関数
    
    Parameters:
    -----------
    eigenvectors : ndarray, shape (2,2)
        2次元の固有ベクトル
    eigenvalues : ndarray, shape (2,)
        2次元の固有値
    mean : ndarray, shape (2,)
        データの平均値 (楕円の中心)
    color : str, optional
        楕円の色
    linestyle : str, optional
        線のスタイル ('-' or '--')
    """
    # 楕円のパラメータ角度
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 単位円上の点を生成
    circle = 2 * np.array([np.cos(theta), np.sin(theta)])
    
    # 固有値で拡大縮小
    scaled = np.sqrt(eigenvalues)[:, np.newaxis] * circle
    
    # 固有ベクトルで回転
    rotated = eigenvectors @ scaled
    
    # 平均値で平行移動
    ellipse = rotated + mean[:, np.newaxis]
    
    # 楕円をプロット
    plt.plot(ellipse[0], ellipse[1], linestyle=linestyle, c=color, alpha=0.9, lw=1)


def plot_velocity_spin_comparison(all_data_1, all_data_2=None):
    """
    1つまたは2つの時期のデータを比較プロットする関数。各選手ごとに比較を行う。
    
    Parameters
    ----------
    all_data_1 : pandas.DataFrame
        1つ目のデータセット
    all_data_2 : pandas.DataFrame, optional
        2つ目のデータセット（オプション）
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap("tab10")

    # 選手リストを取得
    if all_data_2 is not None:
        # 2つのデータセットがある場合は共通の選手を取得
        players = sorted(list(set(all_data_1["player"].unique()) & set(all_data_2["player"].unique())))
    else:
        # 1つのデータセットの場合はそのデータの選手を取得
        players = sorted(list(all_data_1["player"].unique()))

    # 各選手に対して処理
    for i, player in enumerate(all_data_1["player"].unique()):
        # 1つ目のデータセットの処理
        data_1 = all_data_1[all_data_1["player"] == player]
        x_1 = data_1["release_speed"].values
        y_1 = data_1["release_spin_rate"].values

        # 平均値を計算して表示
        print(f"\n{player}:")
        print(f"Dataset 1 - 平均球速: {np.mean(x_1):.1f} km/h, 平均回転数: {np.mean(y_1):.0f} rpm")

        # データ処理
        X_1 = np.column_stack((x_1, y_1))
        pca_1 = PCA(n_components=2)
        pca_1.fit(X_1)

        center_1 = X_1.mean(axis=0)
        vec_1 = pca_1.components_[0]

        # PCA回帰線と楕円を描画
        scale = 30
        line_pca_x_1 = center_1[0] + np.array([-5, 5]) * scale * vec_1[0]
        line_pca_y_1 = center_1[1] + np.array([-5, 5]) * scale * vec_1[1]

        eigenvals = pca_1.explained_variance_
        eigenvecs = pca_1.components_

        plt.plot(line_pca_x_1, line_pca_y_1, linestyle="-", color=colors(i % 10), 
                linewidth=1.5, label=f"{player}_1")
        
        # 1つ目のデータセットの楕円を描画
        ellipse_plot(eigenvecs, eigenvals, center_1, colors(i % 10), linestyle='-')

        # 2つ目のデータセットがある場合の処理
        if all_data_2 is not None:
            data_2 = all_data_2[all_data_2["player"] == player]
            x_2 = data_2["release_speed"].values
            y_2 = data_2["release_spin_rate"].values

            print(f"Dataset 2 - 平均球速: {np.mean(x_2):.1f} km/h, 平均回転数: {np.mean(y_2):.0f} rpm")

            X_2 = np.column_stack((x_2, y_2))
            pca_2 = PCA(n_components=2)
            pca_2.fit(X_2)
            center_2 = X_2.mean(axis=0)
            vec_2 = pca_2.components_[0]

            line_pca_x_2 = center_2[0] + np.array([-5, 5]) * scale * vec_2[0]
            line_pca_y_2 = center_2[1] + np.array([-5, 5]) * scale * vec_2[1]
            plt.plot(line_pca_x_2, line_pca_y_2, linestyle="--", color=colors(i % 10), 
                    linewidth=1.5, label=f"{player}_2")
            
            # 2つ目のデータセットの楕円を描画
            eigenvals_2 = pca_2.explained_variance_
            eigenvecs_2 = pca_2.components_
            ellipse_plot(eigenvecs_2, eigenvals_2, center_2, colors(i % 10), linestyle='--')

    plt.xlabel("Velocity (km/h)", fontsize=12)
    plt.ylabel("Spin Rate (rpm)", fontsize=12)
    plt.title("4-Seam Fastball: Spin Rate vs Velocity Comparison", fontsize=14)
    plt.xlim(140, 165)
    plt.ylim(1800, 2750)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


all_data_202509 = process_data_ff('statcast_data_202509')
all_data_202506 = process_data_ff('statcast_data_202506')

# # 既存のデータを使用して比較プロット
plot_velocity_spin_comparison(all_data_202509, all_data_202506)
