# MLB_statcast_spin_velocity
日本人投手のファストボールと球速と回転数の全投球分布の統計

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
