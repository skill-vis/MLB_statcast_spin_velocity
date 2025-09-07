# MLB_statcast_spin_velocity
日本人投手のファストボールと球速と回転数の全投球分布の統計

## stadtcast.pyについて
適宜，関数内で変更し，自分仕様にしていただきたい．

最後の

all_data_202509 = fetch_and_process_data("statcast_data_202509")
で，データを作る．もしデータが存在するときは，ダウンロードしないで，data_dir内にあるデータを利用する．存在しないときはダウンロードする．

fetch_and_process_data(data_dir, end_date=None)の引数に，

data_dir引数：データ保存ディレクトリ名
end_date引数：データ取得の終了日 (YYYY-MM-DD形式)。指定がない場合は年末まで取得

例
all_data_202506 = fetch_and_process_data("statcast_data_202506", "2025-06-31")

-----
plot_velocity_spin(all_data_202509)

でグラフを描く．

以下は，これを使って描いた２つのグラフをさらに加工

<img width="2211" height="800" alt="Figure_comp" src="https://github.com/user-attachments/assets/6501be7c-4b4e-4ee6-8f36-ca8b8ca6a304" />

**注意**：　同時にこの２つのグラフを描くような仕様ではなく，fetch_and_process_data関数を少し書き換える必要がある．

## stadtcast_ellips.pyについて
stadtcast.pyで作られたデータから，１つまたは２つのデータを与えて，PCAの楕円と傾きを描画（２つの場合は比較）する．

<img width="1200" height="800" alt="Figure_comp2" src="https://github.com/user-attachments/assets/d1137e85-218f-4dcb-943d-650983427132" />

