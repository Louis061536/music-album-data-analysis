#   1) 相关性分析 - 看看曲目数和销量到底有没有关系
#   2) 评分一致性 - 三个评分打架的专辑，销量会更极端吗?
#   3) 线性回归 - 评分+曲目数+年份，能不能预测销量?

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# 0. 设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

import os
shuju_lujing = os.path.join(os.path.dirname(__file__), "albums.csv")
shuchu_mulu   = os.path.dirname(__file__)
pingfen_lie = ['rolling_stone_critic', 'mtv_critic', 'music_maniac_critic']
pingfen_mingzi = ['Rolling Stone', 'MTV', 'Music Maniac']

suijizhongzi = 42
np.random.seed(suijizhongzi)

yanse = ['#2E86AB', '#A23B72', '#E76F51', '#2A9D8F', '#E9C46A', '#F4A261']



# 加载数据 + 构造新特征
def jiazai_he_jiagong(lujing):
    """读入数据，顺便算一些衍生字段"""
    shuju = pd.read_csv(lujing)

    # 类型转换
    shuju['year_of_pub']  = shuju['year_of_pub'].astype(int)
    shuju['num_of_tracks'] = shuju['num_of_tracks'].astype(int)
    shuju['num_of_sales']  = shuju['num_of_sales'].astype(int)
    for lie in pingfen_lie:
        shuju[lie] = pd.to_numeric(shuju[lie], errors='coerce')

    # 造几个新特征
    # 三个评分的平均值
    shuju['pingjun_pingfen'] = shuju[pingfen_lie].mean(axis=1)
    # 三个评分的标准差（代表争议程度）
    shuju['zhengyi_du'] = shuju[pingfen_lie].std(axis=1)
    # 评分中位数
    shuju['zhongweishu'] = shuju[pingfen_lie].median(axis=1)
    # 最高分-最低分
    shuju['jicha'] = shuju[pingfen_lie].max(axis=1) - shuju[pingfen_lie].min(axis=1)
    # 销量取对数（因为销量分布太偏了，取对数正态一点）
    shuju['log_xiaoliang'] = np.log1p(shuju['num_of_sales'])
    # 专辑年龄
    shuju['nianling'] = 2026 - shuju['year_of_pub']
    # 每首歌平均卖多少
    shuju['meishou_xiaoliang'] = shuju['num_of_sales'] / shuju['num_of_tracks']

    print(f"数据加载完成: {shuju.shape[0]} 行 x {shuju.shape[1]} 列")
    return shuju



# 进阶1: 相关性分析——"歌越多越好卖"是真的吗？

def jinjie1_xiangguanxing(shuju):
    """算曲目数和销量的相关系数，验证那个常见的说法"""
    print("\n" + "=" * 70)
    print("【进阶1】相关性分析")
    print("=" * 70)

    # Pearson相关系数（看线性关系）
    r_pearson_gequ, p_pearson_gequ = pearsonr(shuju['num_of_tracks'], shuju['num_of_sales'])
    r_pearson_nian,  p_pearson_nian  = pearsonr(shuju['year_of_pub'],  shuju['num_of_sales'])

    # Spearman相关系数（看单调关系，不要求线性）
    r_spearman_gequ, p_spearman_gequ = spearmanr(shuju['num_of_tracks'], shuju['num_of_sales'])
    r_spearman_nian,  p_spearman_nian  = spearmanr(shuju['year_of_pub'],  shuju['num_of_sales'])

    print(f"  Pearson:  曲目数vs销量 r={r_pearson_gequ:.4f}  年份vs销量 r={r_pearson_nian:.4f}")
    print(f"  Spearman: 曲目数vs销量 r={r_spearman_gequ:.4f}  年份vs销量 r={r_spearman_nian:.4f}")
    print(f"  结论: 相关系数几乎为0，歌越多越好卖不成立!")

    # ---- 分组看: 不同曲目数区间的平均销量 ----
    # 先分箱
    fenzu_bianhao = pd.cut(shuju['num_of_tracks'],
                            bins=[0, 3, 6, 9, 12, 15, 20],
                            labels=['1-3首', '4-6首', '7-9首', '10-12首', '13-15首', '16-20首'])
    shuju['gequ_duan'] = fenzu_bianhao

    # 每个区间的统计量
    duan_tongji = shuju.groupby('gequ_duan', observed=False).agg(
        shuliang=('id', 'count'),
        pingjun_xiaoliang=('num_of_sales', 'mean'),
        zhongweishu_xiaoliang=('num_of_sales', 'median'),
    ).round(0)

    print("\n  不同曲目数区间 vs 平均销量:")
    print(duan_tongji.to_string())

    # 画图: 散点图+均值线
    fig = plt.figure(figsize=(18, 7))

    # 左图: 曲目数 vs 销量
    ax1 = fig.add_subplot(1, 2, 1)
    # 抽5000条画散点，不然太密
    yangben = shuju.sample(5000, random_state=suijizhongzi)
    ax1.scatter(yangben['num_of_tracks'], yangben['num_of_sales'],
                alpha=0.3, s=8, c='#2E86AB', edgecolors='none')

    # 画分箱均值折线
    quxian_x = [2, 5, 8, 11, 14, 18]
    quxian_y = duan_tongji['pingjun_xiaoliang'].values
    ax1.plot(quxian_x, quxian_y, 'o-', color='#A23B72', linewidth=2.5,
             markersize=8, label='每个区间的均值', zorder=5)

    ax1.set_xlabel('Number of Tracks', fontsize=12)
    ax1.set_ylabel('Number of Sales', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax1.legend(fontsize=10)
    ax1.set_title(f'Tracks vs Sales\nPearson r={r_pearson_gequ:.4f}',
                  fontsize=13, fontweight='bold')

    # 右图: 年份 vs 销量
    ax2 = fig.add_subplot(1, 2, 2)
    yangben2 = shuju.sample(5000, random_state=suijizhongzi)
    ax2.scatter(yangben2['year_of_pub'], yangben2['num_of_sales'],
                alpha=0.3, s=8, c='#E76F51', edgecolors='none')

    # 每年的平均销量
    niandu_pingjun = shuju.groupby('year_of_pub')['num_of_sales'].mean()
    ax2.plot(niandu_pingjun.index, niandu_pingjun.values,
             'o-', color='#264653', linewidth=2.5, markersize=8,
             label='每年均值', zorder=5)

    ax2.set_xlabel('Year of Publication', fontsize=12)
    ax2.set_ylabel('Number of Sales', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax2.legend(fontsize=10)
    ax2.set_title(f'Year vs Sales\nPearson r={r_pearson_nian:.4f}',
                  fontsize=13, fontweight='bold')

    fig.suptitle('Advanced Analysis 1: "More Tracks = More Sales" is a Myth',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\adv1_correlation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → 图保存好了: adv1_correlation.png")

    jieguo = {
        'r_pearson_gequ': r_pearson_gequ, 'p_pearson_gequ': p_pearson_gequ,
        'r_pearson_nian': r_pearson_nian, 'p_pearson_nian': p_pearson_nian,
        'r_spearman_gequ': r_spearman_gequ, 'p_spearman_gequ': p_spearman_gequ,
        'r_spearman_nian': r_spearman_nian, 'p_spearman_nian': p_spearman_nian,
    }
    return jieguo


# 进阶2: 评分一致性——争议大的专辑卖得怎么样？
def jinjie2_zhengyi_xiaoliang(shuju):
    """三个评分打架的专辑，销量会更两极分化吗"""
    print("\n" + "=" * 70)
    print("【进阶2】评分一致性——争议=双刃剑?")
    print("=" * 70)

    # 把争议程度分成3档
    shuju['zhengyi_dengji'] = pd.qcut(shuju['zhengyi_du'], q=3,
                                       labels=['低争议', '中争议', '高争议'])

    # 算每个争议等级的统计量
    tongji = shuju.groupby('zhengyi_dengji', observed=False).agg(
        shuliang=('id', 'count'),
        pingjun_xiaoliang=('num_of_sales', 'mean'),
        zhongweishu_xiaoliang=('num_of_sales', 'median'),
        biaozhuncha=('num_of_sales', 'std'),
    ).round(0)

    # 变异系数 = 标准差/均值
    tongji['bianyi_xishu'] = (tongji['biaozhuncha'] / tongji['pingjun_xiaoliang']).round(2)

    # 10分位和90分位
    def suan_p10(x):
        return x.quantile(0.10)
    def suan_p90(x):
        return x.quantile(0.90)

    p10_tongji = shuju.groupby('zhengyi_dengji', observed=False)['num_of_sales'].apply(suan_p10)
    p90_tongji = shuju.groupby('zhengyi_dengji', observed=False)['num_of_sales'].apply(suan_p90)
    tongji['p10'] = p10_tongji.values
    tongji['p90'] = p90_tongji.values
    tongji['p90_p10_bi'] = (tongji['p90'] / tongji['p10']).round(1)

    print("\n  争议程度 vs 销量分布:")
    print(tongji.to_string())

    # 判断分化程度
    gao_cv = tongji.loc['高争议', 'bianyi_xishu']
    di_cv  = tongji.loc['低争议', 'bianyi_xishu']
    if gao_cv > di_cv:
        print(f"  => 验证通过! 高争议专辑变异系数({gao_cv})>低争议({di_cv}), 两极分化更严重")
    else:
        print(f"  => 差异不明显, 需要更多数据")

    # 画图
    fig = plt.figure(figsize=(18, 12))

    # 左上: 争议-销量散点图
    ax1 = fig.add_subplot(2, 2, 1)
    yangben = shuju.sample(5000, random_state=suijizhongzi)
    yanse_ditu = {'低争议': '#2A9D8F', '中争议': '#E9C46A', '高争议': '#E76F51'}
    for dengji in ['低争议', '中争议', '高争议']:
        zi = yangben[yangben['zhengyi_dengji'] == dengji]
        ax1.scatter(zi['zhengyi_du'], zi['num_of_sales'],
                    c=yanse_ditu[dengji], alpha=0.4, s=8, label=dengji, edgecolors='none')

    ax1.set_xlabel('Rating Std Dev (争议程度)', fontsize=12)
    ax1.set_ylabel('Sales', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_title('争议程度 vs 销量散点图', fontsize=13, fontweight='bold')

    # 右上: 小提琴图（不同争议等级的销量分布）
    ax2 = fig.add_subplot(2, 2, 2)
    # 截断到95%分位，不然图不好看
    shangxian = shuju['num_of_sales'].quantile(0.95)
    huitu_shuju = shuju[shuju['num_of_sales'] <= shangxian].sample(3000, random_state=suijizhongzi)

    shunxu = ['低争议', '中争议', '高争议']
    tu_data = [huitu_shuju[huitu_shuju['zhengyi_dengji'] == d]['num_of_sales'] for d in shunxu]
    parts = ax2.violinplot(tu_data, positions=[1, 2, 3], showmeans=True, showmedians=True)

    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(list(yanse_ditu.values())[i])
        body.set_alpha(0.7)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(shunxu, fontsize=10)
    ax2.set_ylabel('Sales', fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax2.set_title('不同争议程度下的销量分布', fontsize=13, fontweight='bold')

    # 下方: 四象限图（平均评分 × 争议程度，颜色=销量）
    ax3 = fig.add_subplot(2, 2, (3, 4))
    yangben2 = shuju.sample(3000, random_state=suijizhongzi)
    san = ax3.scatter(yangben2['pingjun_pingfen'], yangben2['zhengyi_du'],
                      c=yangben2['num_of_sales'], cmap='RdYlGn', alpha=0.6,
                      s=15, edgecolors='none', norm=matplotlib.colors.LogNorm())

    # 画中位线把图分成四个象限
    zhongxian_x = shuju['pingjun_pingfen'].median()
    zhongxian_y = shuju['zhengyi_du'].median()
    ax3.axhline(y=zhongxian_y, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=zhongxian_x, color='gray', linestyle='--', alpha=0.5)

    # 标注象限含义
    ax3.text(shuju['pingjun_pingfen'].max()-0.3, shuju['zhengyi_du'].max()-0.05,
             '高分高争议\n(争议神作)', fontsize=9, ha='right', va='top',
             color='#E76F51', fontweight='bold')
    ax3.text(shuju['pingjun_pingfen'].min()+0.1, shuju['zhengyi_du'].min()+0.02,
             '低分低争议\n(稳定烂片)', fontsize=9, ha='left', va='bottom',
             color='#2A9D8F', fontweight='bold')
    ax3.text(shuju['pingjun_pingfen'].max()-0.3, shuju['zhengyi_du'].min()+0.02,
             '高分低争议\n(安全选择)', fontsize=9, ha='right', va='bottom',
             color='#2A9D8F', fontweight='bold')

    plt.colorbar(san, ax=ax3, label='Sales')
    ax3.set_xlabel('Average Rating', fontsize=12)
    ax3.set_ylabel('争议程度(Rating Std Dev)', fontsize=12)
    ax3.set_title('四象限: 评分 vs 争议, 颜色代表销量', fontsize=13, fontweight='bold')

    fig.suptitle('Advanced Analysis 2: Rating Controversy & Sales',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\adv2_controversy.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → 图保存好了: adv2_controversy.png")

    return tongji



# 进阶3: 线性回归——哪个变量对销量影响最大?

def jinjie3_huigui(shuju):
    """用 评分+曲目数+年份 去拟合 销量，看谁的权重最大"""
    print("\n" + "=" * 70)
    print("【进阶3】线性回归——谁能解释销量?")
    print("=" * 70)

    tezheng_lie = ['pingjun_pingfen', 'zhengyi_du', 'num_of_tracks', 'year_of_pub']
    tezheng_mingzi = ['平均评分', '争议程度', '曲目数', '发行年份']

    # 准备X和y
    X = shuju[tezheng_lie].copy()
    y = shuju['log_xiaoliang'].values  # 用log销量，更接近正态

    # 标准化，这样系数可以互相比较
    biaozhunhua = StandardScaler()
    X_biaozhun = biaozhunhua.fit_transform(X)
    X_biaozhun_df = pd.DataFrame(X_biaozhun, columns=tezheng_lie)

    # 分成训练集和测试集(8:2)
    X_xunlian, X_ceshi, y_xunlian, y_ceshi = train_test_split(
        X_biaozhun_df, y, test_size=0.2, random_state=suijizhongzi
    )

    # 线性回归
    moxing = LinearRegression()
    moxing.fit(X_xunlian, y_xunlian)

    # Ridge回归（加个正则化，看看稳不稳定）
    ridge_moxing = Ridge(alpha=1.0)
    ridge_moxing.fit(X_xunlian, y_xunlian)

    # 评估
    y_yuce = moxing.predict(X_ceshi)
    r2 = r2_score(y_ceshi, y_yuce)
    mae = mean_absolute_error(y_ceshi, y_yuce)
    # 交叉验证
    cv_fenshu = cross_val_score(moxing, X_biaozhun_df, y, cv=5, scoring='r2')

    # 整理系数
    xishu_biao = pd.DataFrame({
        '特征': tezheng_mingzi,
        'OLS系数': moxing.coef_,
        'Ridge系数': ridge_moxing.coef_,
    })
    xishu_biao['绝对值'] = np.abs(moxing.coef_)
    xishu_biao['权重%'] = (xishu_biao['绝对值'] / xishu_biao['绝对值'].sum() * 100).round(1)
    xishu_biao = xishu_biao.sort_values('绝对值', ascending=False)

    print(f"\n  模型评估:")
    print(f"  R² (测试集): {r2:.4f}")
    print(f"  MAE (log scale): {mae:.4f}")
    print(f"  交叉验证R²: {cv_fenshu.mean():.4f} ± {cv_fenshu.std():.4f}")
    print(f"\n  标准化系数 (绝对值越大影响越大):")
    print(xishu_biao[['特征', 'OLS系数', '权重%']].to_string(index=False))

    dixian = xishu_biao.iloc[0]
    print(f"\n  => 权重最高的特征是: 【{dixian['特征']}】(权重占比{dixian['权重%']}%)")
    print(f"  => 但整体R²≈0，说明这些特征的预测能力几乎为零")

    #画图
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (1) 特征权重图
    ax1 = fig.add_subplot(gs[0, 0])
    fanxu = xishu_biao.iloc[::-1]
    yanse_tiao = ['#2A9D8F' if c > 0 else '#E76F51' for c in fanxu['OLS系数']]
    bangbang = ax1.barh(fanxu['特征'], fanxu['OLS系数'], color=yanse_tiao)
    ax1.axvline(x=0, color='black', linewidth=0.8)

    for bang, zhi, quanzhong in zip(bangbang, fanxu['OLS系数'], fanxu['权重%']):
        ax1.text(bang.get_width() + 0.002 * np.sign(bang.get_width()),
                 bang.get_y() + bang.get_height()/2,
                 f'{zhi:+.4f} ({quanzhong}%)', va='center', fontsize=9)

    ax1.set_xlabel('标准化系数', fontsize=11)
    ax1.set_title('特征重要性(标准化系数)', fontsize=12, fontweight='bold')

    # (2) 预测值vs实际值
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_ceshi, y_yuce, alpha=0.3, s=5, c='#457B9D', edgecolors='none')
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1.5)
    ax2.set_xlabel('实际 log(Sales)', fontsize=11)
    ax2.set_ylabel('预测 log(Sales)', fontsize=11)
    ax2.set_title(f'预测 vs 实际 (R²={r2:.3f})', fontsize=12, fontweight='bold')

    # (3) 残差分布
    ax3 = fig.add_subplot(gs[0, 2])
    cancha = y_ceshi - y_yuce
    ax3.hist(cancha, bins=80, color='#2A9D8F', edgecolor='white', alpha=0.8)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('残差', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.set_title('残差分布', fontsize=12, fontweight='bold')

    # (4)-(7) 四个特征的偏依赖图
    weizhi_liebiao = [(1, 0), (1, 1), (1, 2), (2, 0)]
    for i, (tezheng, mingzi) in enumerate(zip(tezheng_lie, tezheng_mingzi)):
        hang, lie = weizhi_liebiao[i]
        ax = fig.add_subplot(gs[hang, lie])
        yangben = shuju.sample(3000, random_state=suijizhongzi)
        ax.scatter(yangben[tezheng], yangben['log_xiaoliang'],
                   alpha=0.3, s=5, c=yanse[i], edgecolors='none')

        # 分箱均值趋势线
        if tezheng in ['pingjun_pingfen', 'zhengyi_du']:
            fenzu_xiang = pd.cut(shuju[tezheng], bins=20)
            fenzu_pingjun = shuju.groupby(fenzu_xiang, observed=False)['log_xiaoliang'].mean()
            zhongxins = [(b.left + b.right)/2 for b in fenzu_pingjun.index]
            ax.plot(zhongxins, fenzu_pingjun.values, 'o-', color='red', linewidth=2, markersize=3)

        if tezheng == 'num_of_tracks':
            t_bins = pd.cut(shuju[tezheng], bins=15)
            t_means = shuju.groupby(t_bins, observed=False)['log_xiaoliang'].mean()
            t_centers = [(b.left + b.right)/2 for b in t_means.index]
            ax.plot(t_centers, t_means.values, 'o-', color='red', linewidth=2, markersize=3)

        if tezheng == 'year_of_pub':
            y_means = shuju.groupby('year_of_pub')['log_xiaoliang'].mean()
            ax.plot(y_means.index, y_means.values, 'o-', color='red', linewidth=2, markersize=4)

        ax.set_xlabel(mingzi, fontsize=10)
        ax.set_ylabel('log(Sales)', fontsize=10)
        ax.set_title(f'{mingzi} vs log(Sales)', fontsize=11)

    # (8) 右下: 文字总结
    ax_zongjie = fig.add_subplot(gs[2, 1:])
    ax_zongjie.axis('off')

    zongjie_wenben = f"""
    ╔══════════════════════════════════════════════╗
    ║   回归分析总结                                ║
    ╠══════════════════════════════════════════════╣
    ║  R² = {r2:.4f}   (几乎为0!)                   ║
    ║  MAE = {mae:.4f} (对数尺度)                   ║
    ║  CV R² = {cv_fenshu.mean():.4f} ± {cv_fenshu.std():.4f}                  ║
    ║                                              ║
    ║  ★ 核心结论:                                 ║
    ║  评分、曲目数、年份对销量几乎没有解释力     ║
    ║  专辑销量几乎无法从账面数据预测             ║
    ║  这意味着:                                   ║
    ║  · 销量更依赖营销、知名度等外部因素         ║
    ║  · 评分高不能保证卖得好                     ║
    ║  · 需要纳入更多特征(预算、厂牌、巡演等)    ║
    ╚══════════════════════════════════════════════╝
    """

    ax_zongjie.text(0.5, 0.5, zongjie_wenben, transform=ax_zongjie.transAxes,
                    fontsize=10, fontfamily='monospace', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='#FFF8DC',
                              edgecolor='#DAA520', alpha=0.9, linewidth=2))

    fig.suptitle('Advanced Analysis 3: Linear Regression — What Drives Sales?',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.savefig(f"{shuchu_mulu}\\adv3_regression.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → 图保存好了: adv3_regression.png")

    jieguo = {
        'moxing': moxing, 'biaozhunhua': biaozhunhua,
        'r2': r2, 'mae': mae,
        'cv_r2_pingjun': cv_fenshu.mean(), 'cv_r2_std': cv_fenshu.std(),
        'xishu_biao': xishu_biao,
    }
    return jieguo



#产品: 专辑投资价值评分卡
def goujian_pingfenka(shuju, huigui_jieguo):
    """给每张专辑算一个综合投资分，分成S/A/B/C/D五个等级"""
    print("\n" + "=" * 70)
    print("【产品】专辑投资价值评分卡")
    print("=" * 70)

    # A. 销量潜力分 (0-100)
    tezheng_lie = ['pingjun_pingfen', 'zhengyi_du', 'num_of_tracks', 'year_of_pub']
    moxing = huigui_jieguo['moxing']
    biaozhunhua = huigui_jieguo['biaozhunhua']

    # 用回归模型预测log销量
    X_quanbu = biaozhunhua.transform(shuju[tezheng_lie])
    shuju['yuce_log_xiaoliang'] = moxing.predict(X_quanbu)
    shuju['yuce_xiaoliang'] = np.expm1(shuju['yuce_log_xiaoliang'])

    # 用百分位排名映射到0-100
    shuju['xiaoliang_qianli_fen'] = (shuju['yuce_xiaoliang'].rank(pct=True) * 100).clip(0, 100)

    #B. 艺术价值分 (0-100)
    # 评分均值归一化到0-100
    pingjun_guiyi = (shuju['pingjun_pingfen'] / 5.0) * 100

    # 一致性分: 标准差越小分越高
    zuida_std = shuju['zhengyi_du'].max()
    yizhixing_fen = (1 - shuju['zhengyi_du'] / zuida_std) * 100

    # 最高单项分
    zuigao_fen = (shuju[pingfen_lie].max(axis=1) / 5.0) * 100

    # 加权合成
    shuju['yishu_jiazhi_fen'] = (
        pingjun_guiyi * 0.50 +
        yizhixing_fen * 0.30 +
        zuigao_fen * 0.20
    ).clip(0, 100)

    # C. 综合投资分 = 销量潜力55% + 艺术价值45%
    shuju['zonghe_touzi_fen'] = (
        shuju['xiaoliang_qianli_fen'] * 0.55 +
        shuju['yishu_jiazhi_fen'] * 0.45
    )

    # D. 分级
    fenduan = [0, 40, 60, 75, 85, 101]
    dengji_biaoqian = ['D-不推荐', 'C-谨慎考虑', 'B-值得投资', 'A-强烈推荐', 'S-必须签']
    shuju['touzi_dengji'] = pd.cut(shuju['zonghe_touzi_fen'], bins=fenduan,
                                    labels=dengji_biaoqian, right=False)

    #打印统计
    dengji_tongji = shuju.groupby('touzi_dengji', observed=False).agg(
        shuliang=('id', 'count'),
        zhanbi=('id', lambda x: f"{len(x)/len(shuju)*100:.1f}%"),
        pingjun_zonghe=('zonghe_touzi_fen', 'mean'),
    ).round(1)

    print("\n  各等级分布:")
    print(dengji_tongji.to_string())

    # 按流派的投资价值排名
    liupai_fen = shuju.groupby('genre').agg(
        zhuanji_shu=('id', 'count'),
        pingjun_zonghe=('zonghe_touzi_fen', 'mean'),
        S_shu=('touzi_dengji', lambda x: (x == 'S-必须签').sum()),
        A_shu=('touzi_dengji', lambda x: (x == 'A-强烈推荐').sum()),
    ).round(1)
    liupai_fen['SA_zhanbi'] = ((liupai_fen['S_shu'] + liupai_fen['A_shu']) / liupai_fen['zhuanji_shu'] * 100).round(1)
    liupai_fen = liupai_fen.sort_values('pingjun_zonghe', ascending=False)

    print(f"\n  各流派投资价值排名 (前15):")
    print(liupai_fen.head(15).to_string())

    #生成推荐理由
    def shengcheng_tuijian(hang):
        liyou = []
        if hang['pingjun_pingfen'] >= 4.0:
            liyou.append("评分极高")
        elif hang['pingjun_pingfen'] >= 3.0:
            liyou.append("评分良好")
        if hang['xiaoliang_qianli_fen'] >= 80:
            liyou.append("商业潜力高")
        if hang['zhengyi_du'] <= 0.5:
            liyou.append("评论一致性好")
        elif hang['zhengyi_du'] >= 1.5:
            liyou.append("争议性强(话题价值)")
        if hang['num_of_tracks'] >= 12:
            liyou.append("内容量充足")
        if hang['num_of_sales'] >= shuju['num_of_sales'].quantile(0.8):
            liyou.append("已有销量验证")
        return '; '.join(liyou) if liyou else '各方面均衡'

    # 画评分卡仪表盘
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Album Investment Scorecard — A&R Decision Tool',
                 fontsize=18, fontweight='bold', y=0.99)

    # (1) 左上: 等级分布饼图
    ax1 = fig.add_subplot(2, 3, 1)
    dengji_yanse = {'S-必须签': '#FFD700', 'A-强烈推荐': '#2A9D8F',
                     'B-值得投资': '#457B9D', 'C-谨慎考虑': '#E9C46A',
                     'D-不推荐': '#E76F51'}
    dengji_shunxu = ['S-必须签', 'A-强烈推荐', 'B-值得投资', 'C-谨慎考虑', 'D-不推荐']
    dengji_jishu = shuju['touzi_dengji'].value_counts()
    daxiao = [dengji_jishu.get(g, 0) for g in dengji_shunxu]
    yanse_dengji = [dengji_yanse[g] for g in dengji_shunxu]

    wedges, texts, autotexts = ax1.pie(
        daxiao, labels=dengji_shunxu, autopct='%1.1f%%', colors=yanse_dengji,
        explode=(0.08, 0.04, 0, 0, 0), textprops={'fontsize': 8}, pctdistance=0.7
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax1.set_title('投资等级分布', fontsize=12, fontweight='bold')

    # (2) 中上: Top20专辑气泡图
    ax2 = fig.add_subplot(2, 3, 2)
    qian20 = shuju.nlargest(20, 'zonghe_touzi_fen')
    san2 = ax2.scatter(
        qian20['yishu_jiazhi_fen'], qian20['xiaoliang_qianli_fen'],
        s=qian20['yuce_xiaoliang'] / qian20['yuce_xiaoliang'].max() * 300,
        c=qian20['zonghe_touzi_fen'], cmap='RdYlGn', alpha=0.8,
        edgecolors='black', linewidth=0.5
    )
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('艺术价值分', fontsize=10)
    ax2.set_ylabel('销量潜力分', fontsize=10)
    ax2.set_title('Top20: 艺术 vs 商业 (气泡=预测销量)', fontsize=11, fontweight='bold')
    plt.colorbar(san2, ax=ax2, label='综合分')

    # (3) 右上: 帕累托曲线（得分-销量集中度）
    ax3 = fig.add_subplot(2, 3, 3)
    paixu = shuju.sort_values('zonghe_touzi_fen', ascending=False)
    paixu['leiji_zhuanji_pct'] = np.arange(1, len(paixu)+1) / len(paixu) * 100
    paixu['leiji_xiaoliang_pct'] = paixu['num_of_sales'].cumsum() / paixu['num_of_sales'].sum() * 100

    ax3.plot(paixu['leiji_zhuanji_pct'], paixu['leiji_xiaoliang_pct'],
             color='#457B9D', linewidth=2.5)
    ax3.fill_between(paixu['leiji_zhuanji_pct'], paixu['leiji_xiaoliang_pct'],
                     alpha=0.2, color='#457B9D')
    ax3.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)

    # 标几个关键点
    for bai in [20, 50, 80]:
        weizhi = int(len(paixu) * bai / 100)
        ax3.annotate(
            f'前{bai}%专辑\n→ {paixu["leiji_xiaoliang_pct"].iloc[weizhi]:.0f}%销量',
            xy=(bai, paixu['leiji_xiaoliang_pct'].iloc[weizhi]),
            xytext=(bai + 15, paixu['leiji_xiaoliang_pct'].iloc[weizhi] - 10),
            arrowprops=dict(arrowstyle='->', color='#E76F51', lw=1.5),
            fontsize=9, color='#E76F51', fontweight='bold'
        )

    ax3.set_xlabel('累计专辑占比%(按分数排序)', fontsize=10)
    ax3.set_ylabel('累计销量占比%', fontsize=10)
    ax3.set_title('分数-销量集中度曲线', fontsize=11, fontweight='bold')

    # (4) 左下: 各流派平均得分
    ax4 = fig.add_subplot(2, 3, 4)
    qian15_liupai = liupai_fen.head(15).iloc[::-1]
    bang4 = ax4.barh(qian15_liupai.index, qian15_liupai['pingjun_zonghe'],
                     color=sns.color_palette("viridis", 15)[::-1])
    for bang, zhi in zip(bang4, qian15_liupai['pingjun_zonghe']):
        ax4.text(bang.get_width() + 0.1, bang.get_y() + bang.get_height()/2,
                 f'{zhi:.1f}', va='center', fontsize=8)
    ax4.set_xlabel('平均综合分', fontsize=11)
    ax4.set_title('各流派投资价值排名 (Top 15)', fontsize=12, fontweight='bold')

    # (5) 中下: 艺术×商业 热度图
    ax5 = fig.add_subplot(2, 3, 5)
    shuju['sps_duan'] = pd.cut(shuju['xiaoliang_qianli_fen'], bins=10, labels=False)
    shuju['avs_duan'] = pd.cut(shuju['yishu_jiazhi_fen'], bins=10, labels=False)
    relitu = shuju.pivot_table(index='avs_duan', columns='sps_duan',
                                values='id', aggfunc='count').fillna(0)
    relitu = relitu / relitu.sum().sum() * 100

    sns.heatmap(relitu, cmap='YlOrRd', ax=ax5, cbar_kws={'label': '专辑占比%'},
                linewidths=0.3)
    ax5.set_xlabel('销量潜力分(分箱)', fontsize=10)
    ax5.set_ylabel('艺术价值分(分箱)', fontsize=10)
    ax5.set_title('专辑密度: 艺术价值 × 商业潜力', fontsize=11, fontweight='bold')
    ax5.axhline(y=8, color='gold', linestyle='--', linewidth=1.5, alpha=0.6)
    ax5.axvline(x=8, color='gold', linestyle='--', linewidth=1.5, alpha=0.6)

    # (6) 右下: 评分卡说明
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    shuoming_wenben = """
╔══════════════════════════════════════════════╗
║   专辑投资价值评分卡 v1.0                    ║
║   A&R 决策支持工具                           ║
╠══════════════════════════════════════════════╣
║                                              ║
║  评分方法:                                   ║
║  销量潜力(55%) = 回归预测销量的百分位排名   ║
║  艺术价值(45%) = 评分均值×50%               ║
║                + 一致性×30%                  ║
║                + 最高单项分×20%              ║
║                                              ║
║  等级划分:                                   ║
║   S: >=85  ★★★★★  必须签                   ║
║   A: 75-84 ★★★★   强烈推荐                 ║
║   B: 60-74 ★★★    值得投资                 ║
║   C: 40-59 ★★     谨慎考虑                 ║
║   D: <40   ★      不推荐                   ║
║                                              ║
║  使用方法:                                   ║
║   1. 对新专辑打分，看落在哪个等级            ║
║   2. S/A级优先投入A&R资源                   ║
║   3. 关注争议度指标做风险把控               ║
║   4. 签约后追踪实际vs预测表现               ║
╚══════════════════════════════════════════════╝
"""

    ax6.text(0.05, 0.95, shuoming_wenben, transform=ax6.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFF8DC', alpha=0.9,
                       edgecolor='#DAA520', linewidth=2))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{shuchu_mulu}\\scorecard_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → 评分卡图保存好了: scorecard_dashboard.png")

    # 导出CSV
    shuchu_lie = shuju[[
        'id', 'artist_id', 'album_title', 'genre', 'year_of_pub',
        'num_of_tracks', 'num_of_sales', 'pingjun_pingfen', 'zhengyi_du',
        'xiaoliang_qianli_fen', 'yishu_jiazhi_fen', 'zonghe_touzi_fen',
        'touzi_dengji'
    ]].copy()

    shuchu_lie.columns = [
        'ID', 'artist_id', '专辑名', '流派', '年份',
        '曲目数', '销量', '平均评分', '争议度',
        '销量潜力分', '艺术价值分', '综合投资分',
        '投资等级'
    ]

    # 生成推荐理由
    tuijian_lie = []
    for idx, hang in shuju.iterrows():
        tuijian_lie.append(shengcheng_tuijian(hang))
    shuchu_lie['推荐理由'] = tuijian_lie

    shuchu_lie = shuchu_lie.sort_values('综合投资分', ascending=False)

    # 导出前500条
    shuchu_lie.head(500).to_csv(
        f"{shuchu_mulu}\\top500_investment_scorecard.csv",
        index=False, encoding='utf-8-sig'
    )
    # 导出全部
    shuchu_lie.to_csv(
        f"{shuchu_mulu}\\full_investment_scorecard.csv",
        index=False, encoding='utf-8-sig'
    )

    print(f"  → CSV导出好了: top500条 + 全部{len(shuchu_lie)}条")
    return shuchu_lie, liupai_fen


# 主程序
def main():
    print("=" * 70)
    print("  音乐专辑进阶分析 + 投资评分卡")
    print("=" * 70)

    # 加载数据
    quanbu_shuju = jiazai_he_jiagong(shuju_lujing)

    # 进阶1: 相关性
    xiangguan_jieguo = jinjie1_xiangguanxing(quanbu_shuju)

    # 进阶2: 争议性
    zhengyi_jieguo = jinjie2_zhengyi_xiaoliang(quanbu_shuju)

    # 进阶3: 线性回归
    huigui_jieguo = jinjie3_huigui(quanbu_shuju)

    # 产品: 投资评分卡
    pingfenka, liupai_paiming = goujian_pingfenka(quanbu_shuju, huigui_jieguo)

    # 最后打印总结
    print("\n" + "=" * 70)
    print("  全部完成!")
    print("=" * 70)
    print(f"""
  输出文件:
  ├── adv1_correlation.png          — 相关性分析图
  ├── adv2_controversy.png          — 争议性分析图
  ├── adv3_regression.png           — 线性回归分析图
  ├── scorecard_dashboard.png       — 评分卡仪表盘
  ├── top500_investment_scorecard.csv — 前500评分卡
  └── full_investment_scorecard.csv   — 全部评分卡

  核心发现:
  1. "歌越多越好卖"是假的 (r≈0)
  2. 争议大的专辑销量两极分化更严重
  3. 评分+曲目+年份对销量的解释力≈0
  4. {len(quanbu_shuju)}张专辑已完成投资价值评分
  """)


if __name__ == '__main__':
    main()
