import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

graph = tf.get_default_graph()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            def generate_next_pv(sv, pv, k1, k2):
                """
                設定値svから変動を含む実際の値pvを求める
                sv：設定値
                pv：1ステップ前の実際の値
                k1：svとpvの差異上限を決める指数
                k2：pvの変動距離を決める指数
                """
                ## pvの変動距離を計算
                ## 変動距離はsvとpvの差異上限値k1*svにk2をかけたものを3σとした正規分布から求める
                
                stepsize_3s=k2*k1*sv#k1*svが差異上限値、これにk2をかけることで3σとする
                stepsize=abs(np.random.randn()*stepsize_3s/3)#σ×正規分布の値の絶対値＝変動距離
                
                ## +/-に変動するかを確率的に決定
                ## 変動確率はsvとpvの差異の関数とする
                
                prov=-0.5/(sv*k1)*abs(pv-sv)+0.5
                
                if prov < 0:
                    prov=0
                if pv-sv >=0:
                    coef=np.random.choice([-1, 1], p=[1-prov, prov])
                else:
                    coef=np.random.choice([-1, 1], p=[prov, 1-prov])
                
                return pv+coef*stepsize

            ## 総括伝熱係数Uの算出
            def calc_U(Th1, wh, wc):
                ch=4200
                cc=2100
                Tc1=60
                Tc2=20
                A=3.0
                q=wc*cc*(Tc1-Tc2)
                delta_Th=q/(wh*ch)
                Th2=95-delta_Th
                T1=Th1-Tc1
                T2=Th2-Tc2
                delta_T_ln=(T1-T2)/np.log(T1/T2)
                return q/(A*delta_T_ln)

            ## 正常時のUの値から異常時のUの値を求める
            def calc_U_with_anomaly(U):
                step=600
                anomaly_effect=[0 if step>i else 0.02*(i-step)**1.4 for i in np.arange(len(U))]
                return U+anomaly_effect


            ## 設定値のリスト（1000時間分）
            Th1_sv=np.random.uniform(88, 92, 1000)
            wh_sv=np.random.uniform(2.4, 2.6, 1000)
            wc_sv=[3.0]*1000

            ##初期値
            Th1_pv=[90]
            wh_pv=[2.5]
            wc_pv=[3.0]

            ## 設定値から揺れを含む実際の値を求める
            for Th1_s, wh_s, wc_s in zip(Th1_sv, wh_sv, wc_sv):
                Th1_p=Th1_pv[-1]
                wh_p=wh_pv[-1]
                wc_p=wc_pv[-1]
                Th1_pv.append(generate_next_pv(Th1_s, Th1_p, 0.05, 0.5))
                wh_pv.append(generate_next_pv(wh_s, wh_p, 0.05, 0.5))
                wc_pv.append(generate_next_pv(wc_s, wc_p, 0.05, 0.5))

            # データをnp.arrayに変換
            Th1_sv=np.array(Th1_sv)
            wh_sv=np.array(wh_sv)
            wc_sv=np.array(wc_sv)
            Th1_pv=np.array(Th1_pv)
            wh_pv=np.array(wh_pv)
            wc_pv=np.array(wc_pv)
            U_pv=calc_U(Th1_pv,wh_pv, wc_pv)
            U_an=calc_U_with_anomaly(U_pv)

            #データの整理
            X_data = Th1_pv.reshape(1, -1) # 2次元配列に変換
            X_data = np.append(X_data, wh_pv.reshape(1, -1), axis=0) # 2次元配列にしてappend
            X_data = np.append(X_data, wc_pv.reshape(1, -1), axis=0)
            X_data=np.transpose(X_data)
            y_data=U_an.reshape(1, -1)
            y_data=np.transpose(y_data)

            #訓練データ（600日目まで）とテストデータ（601日目以降）に分ける
            train_X=X_data[:601]
            train_y=y_data[:601]
            test_X=X_data[601:]
            test_y=y_data[601:]

            #モデルの学習（線形回帰モデル）
            model=LinearRegression()
            model.fit(train_X, train_y)
            pred_y=model.predict(X_data)

            U_actual=y_data
            U_predict=pred_y
            diff_U=U_actual-U_predict
            pred_answer="異常は検出されませんでした（失敗）。"

            for i in np.arange(600, 1000):
                if diff_U[i]>20:
                    pred_answer="異常兆候開始から"+str(i-599)+"日目に異常を検出しました。"
                    break
            
            ## プロット
            fig1=plt.figure(figsize=(3, 2))
            fig2=plt.figure(figsize=(3, 2))
            fig3=plt.figure(figsize=(3, 2))
            fig4=plt.figure(figsize=(3, 2))
            fig5=plt.figure(figsize=(3, 2))

            ax1=fig1.add_subplot(111)
            ax2=fig2.add_subplot(111)
            ax3=fig3.add_subplot(111)
            ax4=fig4.add_subplot(111)
            ax5=fig5.add_subplot(111)

            ax1.set_ylabel("Th1, ℃")
            ax2.set_ylabel("wh, kg/s")
            ax3.set_ylabel("wC, kg/s")
            ax4.set_ylabel("U [J/(m²s K)]")
            ax5.set_ylabel("diff_U [J/(m²s K)]")
            ax1.plot(Th1_pv)
            ax2.plot(wh_pv)
            ax3.plot(wc_pv)
            ax4.plot(U_actual)
            ax4.plot(U_predict)
            ax5.plot(diff_U)
            fig1.savefig("./static/figure1.png")
            fig2.savefig("./static/figure2.png")
            fig3.savefig("./static/figure3.png")
            fig4.savefig("./static/figure4.png")
            fig5.savefig("./static/figure5.png")
            url1="./static/figure1.png"
            url2="./static/figure2.png"
            url3="./static/figure3.png"
            url4="./static/figure4.png"
            url5="./static/figure5.png"
            
            

            return render_template("index.html",answer=pred_answer,figure1=url1,figure2=url2,figure3=url3,figure4=url4,figure5=url5)

        return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)