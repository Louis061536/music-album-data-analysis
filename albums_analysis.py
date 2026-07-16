import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os


# 0. 基本设置
# 让matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 配色
yanse_top5 = sns.color_palette("viridis", 5)
yanse_all  = sns.color_palette("viridis_r", 38)

# 文件路径（自动定位到脚本所在目录）
shuju_lujing = os.path.join(os.path.dirname(__file__), "albums.csv")
shuchu_mulu   = os.path.dirname(__file__)

# 加载数据
def jiazai_shuju(lujing):
    """把csv读进来，顺便把数据类型转换好"""
    shuju = pd.read_csv(lujing)

    # 把该是整数的列转成整数
    shuju['year_of_pub']  = shuju['year_of_pub'].astype(int)
    shuju['num_of_tracks'] = shuju['num_of_tracks'].astype(int)
    shuju['num_of_sales']  = shuju['num_of_sales'].astype(int)

    # 评分列可能有问题，用coerce把坏数据变成NaN
    pingfen_lie = ['rolling_stone_critic', 'mtv_critic', 'music_maniac_critic']
    for lie_ming in pingfen_lie:
        shuju[lie_ming] = pd.to_numeric(shuju[lie_ming], errors='coerce')

    print(f"数据加载好了: {shuju.shape[0]} 行, {shuju.shape[1]} 列")
    print(f"年份: {shuju['year_of_pub'].min()} 到 {shuju['year_of_pub'].max()}")
    print(f"流派数量: {shuju['genre'].nunique()} 种")
    return shuju



# 任务1: 每种流派有多少张专辑
def renwu1_tongji_shuliang(shuju):
    """统计各类型专辑的数量，从多到少排序"""
    # groupby之后算每个组的行数
    tongji = shuju.groupby('genre').size()
    # 变成DataFrame方便操作
    jieguo = pd.DataFrame({'liupai': tongji.index, 'zhuanji_shu': tongji.values})
    # 从多到少排
    jieguo = jieguo.sort_values('zhuanji_shu', ascending=False)
    # 重新编号
    jieguo = jieguo.reset_index(drop=True)
    # 加个排名列（从1开始）
    jieguo.index = jieguo.index + 1
    jieguo.index.name = 'paiming'

    print("\n" + "=" * 60)
    print("【任务1】各类型专辑数量 (前10名)")
    print("=" * 60)
    print(jieguo.head(10).to_string())

    #画图: 横向柱状图
    fig, ax = plt.subplots(figsize=(12, 9))
    # 取前15名，倒过来让最大的在上面
    qian15 = jieguo.head(15).iloc[::-1]
    bangbang = ax.barh(qian15['liupai'], qian15['zhuanji_shu'],
                       color=yanse_all[:15][::-1])

    # 在每个柱子右边标上数字
    for bang, zhi in zip(bangbang, qian15['zhuanji_shu']):
        ax.text(bang.get_width() + 50, bang.get_y() + bang.get_height()/2,
                str(zhi), va='center', fontsize=9)

    ax.set_xlabel('Album Count', fontsize=12)
    ax.set_title('Task 1: Number of Albums by Genre (Top 15)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\task1_genre_count.png", dpi=150)
    plt.close(fig)
    print("→ 图保存好了: task1_genre_count.png")

    return jieguo

# 任务2: 每种流派的总销量
def renwu2_tongji_xiaoliang(shuju):
    """统计各类型专辑的销量总数，从多到少排序"""
    # groupby流派，然后对销量求和
    fenzu = shuju.groupby('genre')['num_of_sales'].sum()
    jieguo = pd.DataFrame({'liupai': fenzu.index, 'zong_xiaoliang': fenzu.values})
    jieguo = jieguo.sort_values('zong_xiaoliang', ascending=False)
    jieguo = jieguo.reset_index(drop=True)
    jieguo.index = jieguo.index + 1
    jieguo.index.name = 'paiming'

    # 加一列格式化好的销量（方便看）
    def geshihua_xiaoliang(x):
        return f"{x:,.0f}"
    jieguo['xianshi'] = jieguo['zong_xiaoliang'].apply(geshihua_xiaoliang)

    print("\n" + "=" * 60)
    print("【任务2】各类型专辑销量总数 (前10名)")
    print("=" * 60)
    print(jieguo.head(10)[['liupai', 'xianshi']].to_string())

    #画图
    fig, ax = plt.subplots(figsize=(12, 9))
    qian15 = jieguo.head(15).iloc[::-1]
    bangbang = ax.barh(qian15['liupai'], qian15['zong_xiaoliang'],
                       color=yanse_all[:15][::-1])

    for bang, zhi in zip(bangbang, qian15['zong_xiaoliang']):
        ax.text(bang.get_width() + 5000, bang.get_y() + bang.get_height()/2,
                f"{zhi:,.0f}", va='center', fontsize=8)

    ax.set_xlabel('Total Sales', fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title('Task 2: Total Sales by Genre (Top 15)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\task2_genre_sales.png", dpi=150)
    plt.close(fig)
    print("→ 图保存好了: task2_genre_sales.png")

    return jieguo


# 任务3: 每年发了多少专辑、多少单曲
def renwu3_niandu_tongji(shuju):
    """每年发行的专辑数量和单曲数量"""
    # 按年份分组
    fenzu = shuju.groupby('year_of_pub')
    # 统计两个东西: 专辑数(count id), 单曲总数(sum num_of_tracks)
    zhuanji_shu = fenzu['id'].count()
    danqu_shu = fenzu['num_of_tracks'].sum()

    jieguo = pd.DataFrame({
        'nianfen': zhuanji_shu.index,
        'zhuanji_shu': zhuanji_shu.values,
        'danqu_shu': danqu_shu.values
    })

    print("\n" + "=" * 60)
    print("【任务3】近20年每年专辑 & 单曲数量")
    print("=" * 60)
    print(jieguo.to_string(index=False))

    #画图: 双轴柱状图
    fig, ax1 = plt.subplots(figsize=(14, 7))

    yanse_zhuanji = '#2E86AB'
    yanse_danqu = '#A23B72'

    # 左轴: 专辑数量
    ax1.bar(jieguo['nianfen'] - 0.2, jieguo['zhuanji_shu'], width=0.4,
            color=yanse_zhuanji, alpha=0.85, label='Album Count')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Album Count', fontsize=12, color=yanse_zhuanji)
    ax1.tick_params(axis='y', labelcolor=yanse_zhuanji)
    ax1.set_xticks(jieguo['nianfen'])
    ax1.set_xticklabels(jieguo['nianfen'], rotation=45)

    # 右轴: 单曲数量
    ax2 = ax1.twinx()
    ax2.bar(jieguo['nianfen'] + 0.2, jieguo['danqu_shu'], width=0.4,
            color=yanse_danqu, alpha=0.85, label='Track Count')
    ax2.set_ylabel('Track Count', fontsize=12, color=yanse_danqu)
    ax2.tick_params(axis='y', labelcolor=yanse_danqu)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax1.set_title('Task 3: Yearly Album & Track Count (2000-2019)',
                  fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\task3_yearly_stats.png", dpi=150)
    plt.close(fig)
    print("→ 图保存好了: task3_yearly_stats.png")

    return jieguo


# 任务4: 销量前5的流派，每年卖了多少
def renwu4_top5_niandu_xiaoliang(shuju, xiaoliang_tongji):
    """找到总销量最高的5个流派，看它们每年的销量变化"""
    # 先找到top5流派
    top5_liupai = xiaoliang_tongji.head(5)['liupai'].tolist()
    print(f"\n总销量 Top 5 流派: {top5_liupai}")

    # 只保留top5流派的数据
    top5_shuju = shuju[shuju['genre'].isin(top5_liupai)]

    # 按流派+年份分组求和
    fenzu = top5_shuju.groupby(['genre', 'year_of_pub'])['num_of_sales'].sum()
    fenzu = fenzu.reset_index()
    fenzu.columns = ['liupai', 'nianfen', 'xiaoliang']

    # 做成透视表（每行一年，每列一个流派）
    toushibiao = fenzu.pivot(index='nianfen', columns='liupai', values='xiaoliang')
    toushibiao = toushibiao[top5_liupai]

    print("\n" + "=" * 60)
    print("【任务4】Top 5 流派各年份销量")
    print("=" * 60)
    print(toushibiao.to_string())

    #画折线图
    fig, ax = plt.subplots(figsize=(14, 7))

    for liupai, yanse in zip(top5_liupai, yanse_top5):
        # 取出这个流派的数据
        liupai_shuju = fenzu[fenzu['liupai'] == liupai]
        ax.plot(liupai_shuju['nianfen'], liupai_shuju['xiaoliang'],
                marker='o', linewidth=2, markersize=5,
                color=yanse, label=liupai)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Total Sales', fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    # 设置x轴刻度
    suoyou_nianfen = sorted(fenzu['nianfen'].unique())
    ax.set_xticks(suoyou_nianfen)
    ax.set_xticklabels(suoyou_nianfen, rotation=45)
    ax.legend(title='Genre', fontsize=10, title_fontsize=11)
    ax.set_title('Task 4: Yearly Sales of Top 5 Genres',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\task4_top5_yearly_sales.png", dpi=150)
    plt.close(fig)
    print("→ 图保存好了: task4_top5_yearly_sales.png")

    return toushibiao


# 任务5: 销量前5的流派，三个评分机构的平均分
def renwu5_top5_pingfen(shuju, xiaoliang_tongji):
    """Top 5 流派在三个评分体系里的平均分"""
    top5_liupai = xiaoliang_tongji.head(5)['liupai'].tolist()

    pingfen_lie = ['rolling_stone_critic', 'mtv_critic', 'music_maniac_critic']
    pingfen_mingzi = ['Rolling Stone', 'MTV', 'Music Maniac']

    # 只保留top5的数据
    top5_shuju = shuju[shuju['genre'].isin(top5_liupai)]
    # 按流派分组，算三个评分的平均值
    jieguo = top5_shuju.groupby('genre')[pingfen_lie].mean()
    jieguo = jieguo.round(2)
    jieguo = jieguo.loc[top5_liupai]
    jieguo.columns = pingfen_mingzi

    print("\n" + "=" * 60)
    print("【任务5】Top 5 流派在不同评分体系中的平均评分")
    print("=" * 60)
    print(jieguo.to_string())

    #画分组柱状图
    x = np.arange(len(top5_liupai))
    kuan = 0.25  # 每个柱子的宽度
    yanse_lie = ['#E63946', '#457B9D', '#2A9D8F']

    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(3):
        # 第i个评分体系的柱子位置
        weizhi = x + (i - 1) * kuan
        bangbang = ax.bar(weizhi, jieguo[pingfen_mingzi[i]], kuan,
                          label=pingfen_mingzi[i],
                          color=yanse_lie[i], alpha=0.9,
                          edgecolor='white', linewidth=0.5)

        # 在每个柱子上方标数字
        for bang in bangbang:
            gaodu = bang.get_height()
            ax.text(bang.get_x() + bang.get_width()/2, gaodu + 0.02,
                    f"{gaodu:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(top5_liupai, fontsize=11)
    ax.set_ylabel('Average Rating', fontsize=12)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)
    ax.set_title('Task 5: Average Ratings of Top 5 Genres Across Critics',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{shuchu_mulu}\\task5_top5_avg_ratings.png", dpi=150)
    plt.close(fig)
    print("→ 图保存好了: task5_top5_avg_ratings.png")

    return jieguo


# 任务6: 画一个大拼盘(仪表盘)，把前面结果放一起
def huizong_yibiao(shuju, shuliang_df, xiaoliang_df, niandu_df,
                   top5_niandu, top5_pingfen):
    """把前面几个任务的图拼成一个大图"""
    top5 = xiaoliang_df.head(5)['liupai'].tolist()

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('Music Album Data Analysis Dashboard (2000-2019, 100k albums)',
                 fontsize=18, fontweight='bold', y=0.98)

    #(1) 左上: 各流派专辑数量饼图
    ax1 = fig.add_subplot(2, 3, 1)
    qian10_shuliang = shuliang_df.head(10).copy()
    qita_shuliang = shuliang_df.iloc[10:]['zhuanji_shu'].sum()
    # 把前10之外的合并成"其他"
    xin_hang = pd.DataFrame([{'liupai': 'Others (28 genres)', 'zhuanji_shu': qita_shuliang}])
    qian10_shuliang = pd.concat([qian10_shuliang, xin_hang], ignore_index=True)

    yanse_bing = sns.color_palette("viridis", len(qian10_shuliang))
    wedges, texts, autotexts = ax1.pie(
        qian10_shuliang['zhuanji_shu'], labels=qian10_shuliang['liupai'],
        autopct='%1.1f%%', colors=yanse_bing, pctdistance=0.8,
        textprops={'fontsize': 7}
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax1.set_title('Album Count by Genre (Top 10 + Others)',
                  fontsize=12, fontweight='bold')

    #(2) 中上: 各流派总销量饼图
    ax2 = fig.add_subplot(2, 3, 2)
    qian10_xiaoliang = xiaoliang_df.head(10).copy()
    qita_xiaoliang = xiaoliang_df.iloc[10:]['zong_xiaoliang'].sum()
    xin_hang2 = pd.DataFrame([{'liupai': 'Others (28 genres)', 'zong_xiaoliang': qita_xiaoliang}])
    qian10_xiaoliang = pd.concat([qian10_xiaoliang, xin_hang2], ignore_index=True)

    wedges2, texts2, autotexts2 = ax2.pie(
        qian10_xiaoliang['zong_xiaoliang'], labels=qian10_xiaoliang['liupai'],
        autopct='%1.1f%%', colors=yanse_bing, pctdistance=0.8,
        textprops={'fontsize': 7}
    )
    for t in autotexts2:
        t.set_fontsize(7)
    ax2.set_title('Total Sales by Genre (Top 10 + Others)',
                  fontsize=12, fontweight='bold')

    # (3) 右上: 年度专辑和单曲趋势
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(niandu_df['nianfen'], niandu_df['zhuanji_shu'],
             'o-', color='#2E86AB', linewidth=2, markersize=4, label='Albums')
    ax3_shuang = ax3.twinx()
    ax3_shuang.plot(niandu_df['nianfen'], niandu_df['danqu_shu'],
                    's--', color='#A23B72', linewidth=2, markersize=4, label='Tracks')
    ax3.set_xlabel('Year', fontsize=9)
    ax3.set_ylabel('Albums', fontsize=9, color='#2E86AB')
    ax3_shuang.set_ylabel('Tracks', fontsize=9, color='#A23B72')
    ax3.tick_params(axis='y', labelcolor='#2E86AB')
    ax3_shuang.tick_params(axis='y', labelcolor='#A23B72')
    ax3.set_title('Yearly Albums & Tracks', fontsize=12, fontweight='bold')

    # (4) 左下: Top5年度销量折线
    ax4 = fig.add_subplot(2, 3, 4)
    top5_shuju = shuju[shuju['genre'].isin(top5)]
    top5_fenzu = top5_shuju.groupby(['genre', 'year_of_pub'])['num_of_sales'].sum()
    top5_fenzu = top5_fenzu.reset_index()

    for liupai, yanse in zip(top5, yanse_top5):
        zi = top5_fenzu[top5_fenzu['genre'] == liupai]
        ax4.plot(zi['year_of_pub'], zi['num_of_sales'],
                 marker='.', linewidth=1.5, markersize=3, color=yanse, label=liupai)

    ax4.set_xlabel('Year', fontsize=9)
    ax4.set_ylabel('Total Sales', fontsize=9)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax4.legend(fontsize=7)
    ax4.set_title('Top 5 Genres: Yearly Sales', fontsize=12, fontweight='bold')

    # (5) 中下: 评分热力图
    ax5 = fig.add_subplot(2, 3, 5)
    pingfen_mingzi = ['Rolling Stone', 'MTV', 'Music Maniac']
    sns.heatmap(top5_pingfen[pingfen_mingzi], annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=0, vmax=5, linewidths=0.5,
                ax=ax5, cbar_kws={'label': 'Rating'})
    ax5.set_title('Top 5 Genres: Avg Ratings Heatmap',
                  fontsize=12, fontweight='bold')
    ax5.set_ylabel('Genre', fontsize=10)

    # (6) 右下: 数据摘要
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    zhaiyao_neirong = [
        "=" * 45,
        "   DATA SUMMARY",
        "=" * 45,
        f"  Total Albums:        {len(shuju):,}",
        f"  Total Genres:        {shuju['genre'].nunique()}",
        f"  Year Range:          {shuju['year_of_pub'].min()} - {shuju['year_of_pub'].max()}",
        f"  Total Sales:         {shuju['num_of_sales'].sum():,.0f}",
        f"  Total Tracks:        {shuju['num_of_tracks'].sum():,}",
        f"  Avg Sales/Album:     {shuju['num_of_sales'].mean():,.0f}",
        f"  Avg Tracks/Album:    {shuju['num_of_tracks'].mean():.1f}",
        "",
        "  TOP 5 GENRES BY SALES:",
    ]
    for i, g in enumerate(top5, 1):
        zongliang = xiaoliang_df[xiaoliang_df['liupai'] == g]['zong_xiaoliang'].values[0]
        zhaiyao_neirong.append(f"    {i}. {g:<20s} {zongliang:>12,.0f}")

    zhaiyao_neirong += [
        "",
        "  RATING SYSTEMS:",
        "    Rolling Stone Critic",
        "    MTV Critic",
        "    Music Maniac Critic",
    ]

    ax6.text(0.05, 0.95, '\n'.join(zhaiyao_neirong), transform=ax6.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{shuchu_mulu}\\dashboard.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("→ 仪表盘图保存好了: dashboard.png")


# 主程序
def main():
    print("=" * 60)
    print("  音乐专辑数据分析")
    print("=" * 60)

    # 第一步: 加载数据
    quanbu_shuju = jiazai_shuju(shuju_lujing)

    # 依次做5个任务
    shuliang_jieguo = renwu1_tongji_shuliang(quanbu_shuju)
    xiaoliang_jieguo = renwu2_tongji_xiaoliang(quanbu_shuju)
    niandu_jieguo = renwu3_niandu_tongji(quanbu_shuju)
    top5_niandu_jieguo = renwu4_top5_niandu_xiaoliang(quanbu_shuju, xiaoliang_jieguo)
    top5_pingfen_jieguo = renwu5_top5_pingfen(quanbu_shuju, xiaoliang_jieguo)

    # 最后画个汇总仪表盘
    huizong_yibiao(quanbu_shuju, shuliang_jieguo, xiaoliang_jieguo,
                   niandu_jieguo, top5_niandu_jieguo, top5_pingfen_jieguo)

    print("\n" + "=" * 60)
    print("  全部跑完了! 图片都保存好了!")
    print(f"  输出目录: {shuchu_mulu}")
    print("=" * 60)


if __name__ == '__main__':
    main()
