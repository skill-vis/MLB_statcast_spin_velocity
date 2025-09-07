import os
from pybaseball import statcast_pitcher
from pybaseball.playerid_lookup import playerid_lookup
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def ellipse_plot(eigenvectors, eigenvalues, mean, color): # plot_ellipse
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
    alpha : float, optional
        楕円の透明度
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
    plt.plot(ellipse[0], ellipse[1], '--', c=color, alpha=0.9, lw=1)


def fetch_and_process_data(data_dir, end_date=None):
    """
    指定されたディレクトリに選手データを保存・読み込みし、処理を行う関数
    
    Parameters
    ----------
    data_dir : str
        データを保存するディレクトリパス
    end_date : str, optional
        データ取得の終了日 (YYYY-MM-DD形式)。指定がない場合は年末まで取得
        
    Returns
    -------
    pandas.DataFrame
        処理済みの全選手データ
    """
    # スクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # データディレクトリのパスを作成
    data_path = os.path.join(script_dir, data_dir)
    
    # データディレクトリの確認と作成
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # 対象選手（First Name, Year）
    players = {
        "Yamamoto": ("Yoshinobu", "2025"),
        "Sasaki": ("Roki", "2025"),
        "Imanaga": ("Shota", "2025"),
        "Senga": ("Kodai", "2025"),
        "Matsui": ("Yuki", "2025"),
        "Sugano": ("Tomoyuki", "2025"),
        "Kikuchi": ("Yusei", "2025"),

        "Darvish": ("Yu", "2025"), # 2025年データ
        "Ohtani": ("Shohei", "2025") # 2025年データ
    }

    # データ格納
    all_data = pd.DataFrame()

    # 各選手のデータ取得（2023〜2025）
    for last_name, (first_name, year) in players.items():
        file_path = os.path.join(data_path, f"{last_name}_{year}.csv")
        
        # ファイルが存在する場合はロード、なければダウンロード
        if os.path.exists(file_path):
            print(f"File exists. Loading {last_name} ({year}) data from file")
            df = pd.read_csv(file_path)
        else:
            try:
                pid = playerid_lookup(last_name.replace("_2023", ""), first_name).iloc[0]["key_mlbam"]
                print(f"Fetching {last_name} ({year}) - ID: {pid}")
                end_date_str = end_date if end_date else f"{year}-12-31"
                df = statcast_pitcher(f"{year}-01-01", end_date_str, pid)
                # データを保存
                df.to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error with {last_name}: {e}")
                continue
        
        # データの加工
        df = df[df["pitch_type"] == "FF"]
        df = df[df["release_spin_rate"] > 1800]  # 回転数フィルタ
        df = df[['release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'release_pos_x', 'release_pos_z', 'arm_angle', 'spin_axis', 'vx0', 'vy0', 'vz0', 'p_throws']].dropna()

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
        df["player"] = last_name
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    return all_data


def plot_velocity_spin(all_data):
    """
    球速と回転数の関係をプロットする関数
    
    Parameters
    ----------
    all_data : pandas.DataFrame
        fetch_and_process_data関数で取得した投手データ
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap("tab10")

    for i, player in enumerate(all_data["player"].unique()):
        subset = all_data[all_data["player"] == player]
        x = subset["release_speed"].values
        y = subset["release_spin_rate"].values
        
        # 平均値を計算して表示
        mean_speed = np.mean(x)
        mean_spin = np.mean(y)
        print(f"\n{player}:")
        print(f"平均球速: {mean_speed:.1f} km/h")
        print(f"平均回転数: {mean_spin:.0f} rpm")
        
        plt.scatter(x, y, alpha=0.9, color=colors(i % 10), s=2.5)

        # 最小二乗法回帰
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        plt.plot(x_line, y_line, linestyle="--", color=colors(i % 10), linewidth=1.5)

        # PCA回帰線
        X = np.column_stack((x, y))
        pca = PCA(n_components=2)
        pca.fit(X)
        center = X.mean(axis=0)
        vec = pca.components_[0]
        scale = 30
        line_pca_x = center[0] + np.array([-5, 5]) * scale * vec[0]
        line_pca_y = center[1] + np.array([-5, 5]) * scale * vec[1]
        plt.plot(line_pca_x, line_pca_y, linestyle="-", color=colors(i % 10), linewidth=1.5, label=f"{player}")
        print(f"Explained variance: {pca.explained_variance_}")

        # 全選手の信頼楕円を描画
        eigenvals = pca.explained_variance_
        eigenvecs = pca.components_
        ellipse_plot(eigenvecs, eigenvals, center, colors(i % 10))
        # # 楕円の角度を計算
        # angle = np.arctan2(eigenvecs[0][1], eigenvecs[0][0])
        
        # # 楕円を描画（2σ = 95%信頼区間）
        # ellip = Ellipse(xy=center, width=4*np.sqrt(eigenvals[0]), height=4*np.sqrt(eigenvals[1]),
        #                angle=np.degrees(angle), facecolor='none', edgecolor=colors(i % 10),
        #                linestyle=':', linewidth=1.5, alpha=0.8)
        # plt.gca().add_patch(ellip)

    plt.xlabel("Velocity (km/h)", fontsize=12)
    plt.ylabel("Spin Rate (rpm)", fontsize=12)
    plt.title("4-Seam Fastball: Spin Rate vs Velocity (>1800 rpm)", fontsize=14)
    plt.xlim(140, 165)  # 球速の範囲を固定
    plt.ylim(1800, 2750)  # 回転数の範囲を固定
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

all_data_202509 = fetch_and_process_data("statcast_data_202509")

plot_velocity_spin(all_data_202509)