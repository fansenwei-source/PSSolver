"""Render the Chinese Beris--Edwards derivation to a vector PDF.

The canonical editable source is beris_edwards_stokes_derivation_zh.tex.
This fallback renderer is used because this machine currently has no TeX
engine installed.  Mathematical expressions are rendered by Matplotlib's
TeX-compatible mathtext engine and Chinese text uses the local Noto CJK font.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "beris_edwards_stokes_derivation_zh.pdf"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
ZH = FontProperties(fname=FONT_REGULAR)
ZH_BOLD = FontProperties(fname=FONT_BOLD)

mpl.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


def paragraph(text: str):
    return ("p", text)


def bullet(text: str):
    return ("b", text)


def math(expr: str, size: float = 13.0):
    return ("m", expr, size)


def subhead(text: str):
    return ("h", text)


CHAPTERS = [
    (
        "结论与文献背景",
        [
            paragraph(
                "本文保存此前关于 S、H、Beris–Edwards 应力、能量验号、Stokes "
                "化简及代码核验的数学内容。结论：当前 one-constant 应力公式和"
                "五分量实现正确；主要问题属于模型近似、零模和边界条件，而不是"
                "张量代数错误。"
            ),
            paragraph(
                "2018 年 Shendruk 论文只在正文说明广义应力含 viscous、elastic、"
                "active，并给出主动应力 −ζQ。完整被动应力来自其文献 32 "
                "（Marenduzzo 等，2007）及文献 33（Beris–Edwards 专著）。"
            ),
            math(r"\Pi^{\rm a}_{ij}=-\zeta Q_{ij}"),
            math(
                r"\Pi^{\rm nem}_{ij}=2\lambda M_{ij}(Q:H)"
                r"-\lambda(H_{ik}M_{kj}+M_{ik}H_{kj})"
            ),
            math(
                r"\qquad +(QH-HQ)_{ij}-L_1(\partial_iQ_{kl})(\partial_jQ_{kl})"
                r"-\zeta Q_{ij}",
                12.5,
            ),
            math(r"M=Q+\frac{I}{3},\qquad \beta\alpha=-\zeta"),
            paragraph(
                "代码取 β=−1、α=ζ，因此 βαQ=−ζQ。ζ>0 对应 extensile，ζ<0 "
                "对应 contractile。"
            ),
        ],
    ),
    (
        "序参量、速度梯度与指标约定",
        [
            math(
                r"Q_{ij}=\frac{3S}{2}\left(n_i n_j-\frac{\delta_{ij}}{3}\right)"
                r"=q\left(n_i n_j-\frac{\delta_{ij}}{3}\right),\quad q=\frac{3S}{2}",
                12.5,
            ),
            paragraph(
                "Q 为对称无迹张量。Shendruk 有序区 S_eq=1/3，所以 q_eq=1/2。"
            ),
            math(
                r"W_{ij}=\partial_j u_i,\qquad E=\frac{W+W^T}{2},"
                r"\qquad \Omega=\frac{W-W^T}{2}"
            ),
            math(r"(\nabla\!\cdot\Pi)_i=\partial_j\Pi_{ij}"),
            paragraph(
                "本文所有交换子符号都依赖这一成套约定。若把 W、Ω 或散度定义"
                "成转置版本，QH−HQ 的表面符号会改变，不能只改其中一项。"
            ),
            subhead("Beris–Edwards 共转/流动取向项"),
            math(r"M=Q+\frac{I}{3}"),
            math(
                r"\mathcal{S}(W,Q)=(\lambda E+\Omega)M+M(\lambda E-\Omega)"
                r"-2\lambda M(Q:E)",
                12.4,
            ),
            paragraph(
                "这里的共转项是二阶张量，不是标量序参量 S。不可压缩时 Tr E=0；"
                "又因 Q 对称，所以 Q:W=Q:E。"
            ),
        ],
    ),
    (
        "自由能、分子场与 Q 方程",
        [
            math(
                r"F=\int_V\left[\frac{A}{2}Q_{ij}Q_{ji}"
                r"+\frac{B}{3}Q_{ij}Q_{jk}Q_{ki}"
                r"+\frac{C}{4}(Q_{ij}Q_{ji})^2\right],\mathrm{d}V",
                12.1,
            ),
            math(
                r"\qquad +\int_V\frac{L_1}{2}(\partial_kQ_{ij})(\partial_kQ_{ij})"
                r",\mathrm{d}V",
                12.3,
            ),
            math(r"H=-\left[\frac{\delta F}{\delta Q}\right]^{\rm ST}"),
            math(
                r"H_{ij}=-AQ_{ij}-B\left(Q_{ik}Q_{kj}"
                r"-\frac{\delta_{ij}}{3}Q_{kl}Q_{lk}\right)"
            ),
            math(r"\qquad -C(Q:Q)Q_{ij}+L_1\nabla^2Q_{ij}"),
            math(
                r"(\partial_t+\mathbf{u}\!\cdot\!\nabla)Q-\mathcal{S}"
                r"=\frac{H}{\gamma}"
            ),
            paragraph(
                "关键区别：Q 演化的弛豫项使用 H/γ；反应应力必须使用 raw H。"
                "若应力里再除一次 γ，会改变被动能量配对和应力尺度。"
            ),
        ],
    ),
    (
        "反应应力的能量验号",
        [
            math(
                r"\mathcal{S}=\lambda(EM+ME)-2\lambda M(Q:E)"
                r"+\Omega M-M\Omega"
            ),
            paragraph(
                "利用 Q、H、M、E 对称，Q、H 无迹，Ω 反对称，以及迹的循环"
                "不变性，可把分子场与共转项的双缩并展开为"
            ),
            math(
                r"H:\mathcal{S}=\lambda(HM+MH):E-2\lambda(Q:H)M:E"
                r"-(QH-HQ):\Omega",
                12.4,
            ),
            math(
                r"\Pi^R=2\lambda M(Q:H)-\lambda(HM+MH)+QH-HQ"
            ),
            math(
                r"\Pi^R:W=\left[2\lambda M(Q:H)-\lambda(HM+MH)\right]:E"
                r"+(QH-HQ):\Omega",
                12.2,
            ),
            math(r"H:\mathcal{S}+\Pi^R:W=0", 15),
            paragraph(
                "这个局部恒等式验证两个 −λ 项和 +(QH−HQ) 的符号。QH−HQ "
                "是反对称应力，负责取向与旋转流动之间的可逆交换，不能删除、"
                "翻号或把总应力强制对称化。"
            ),
        ],
    ),
    (
        "形变应力、各向同性压力与主动应力",
        [
            subhead("一般形变应力"),
            math(
                r"\Pi^d_{ij}=-\partial_iQ_{kl}"
                r"\frac{\partial f}{\partial(\partial_jQ_{kl})}"
            ),
            math(
                r"f_{\rm el}=\frac{L_1}{2}(\partial_mQ_{kl})(\partial_mQ_{kl})"
            ),
            math(
                r"\frac{\partial f_{\rm el}}{\partial(\partial_jQ_{kl})}"
                r"=L_1\partial_jQ_{kl}"
            ),
            math(r"\Pi^d_{ij}=-L_1(\partial_iQ_{kl})(\partial_jQ_{kl})", 14),
            paragraph(
                "因此 one-constant 项没有遗漏 1/2 或因子 2。文献有时另外写"
                " fδ_ij；其散度是标量梯度，可并入不可压缩压力。"
            ),
            subhead("主动应力"),
            math(r"\Pi^a=-\zeta Q,\qquad \beta\alpha=-\zeta"),
            paragraph(
                "若 α 本身允许带正负号，就只检查乘积 βα，不应再重复翻转"
                " extensile/contractile 的符号。"
            ),
        ],
    ),
    (
        "黏性应力、Stokes 方程与能量平衡",
        [
            math(r"\Pi^{\rm tot}=-pI+2\eta E+\Pi^R+\Pi^d+\Pi^a"),
            math(
                r"\partial_j(2\eta E_{ij})=\eta\partial_j(\partial_j u_i"
                r"+\partial_i u_j)=\eta\nabla^2u_i"
            ),
            paragraph("上式要求 η 为常数且 ∇·u=0；变黏度时必须保留 ∇·(2ηE)。"),
            math(
                r"0=-\nabla p+\eta\nabla^2\mathbf{u}-\mathrm{fric}\,\mathbf{u}"
                r"+\nabla\!\cdot\Pi^{\rm nem},\qquad \nabla\!\cdot\mathbf{u}=0",
                12.2,
            ),
            subhead("整体能量检查"),
            math(
                r"\frac{\mathrm{d}}{\mathrm{d}t}\left[\int_V\frac{\rho|\mathbf{u}|^2}{2}"
                r",\mathrm{d}V+F\right]"
            ),
            math(
                r"=-\int_V\left[2\eta E:E+\mathrm{fric}|\mathbf{u}|^2"
                r"+\frac{1}{\gamma}H:H\right],\mathrm{d}V"
            ),
            math(r"\mathcal{P}_{\rm active}=\int_V\zeta Q:E,\mathrm{d}V"),
            paragraph(
                "被动部分非增；主动功率没有固定符号，是主动组分向流动注入"
                "能量的渠道。边界条件若做功，还需补上相应边界通量。"
            ),
        ],
    ),
    (
        "五分量表示与代码代数核验",
        [
            math(
                r"(Q_{xx},Q_{xy},Q_{xz},Q_{yy},Q_{yz}),\qquad Q_{zz}=-Q_{xx}-Q_{yy}"
            ),
            math(
                r"Q:H=Q_{xx}H_{xx}+Q_{yy}H_{yy}+Q_{zz}H_{zz}"
            ),
            math(
                r"\qquad +2(Q_{xy}H_{xy}+Q_{xz}H_{xz}+Q_{yz}H_{yz})"
            ),
            paragraph("梯度双缩并也必须给三个非对角分量权重 2；当前 helper 正确。"),
            math(
                r"2\lambda M(Q:H)+(1-\lambda)MH-(1+\lambda)HM"
            ),
            math(
                r"=2\lambda M(Q:H)-\lambda(MH+HM)+(MH-HM)"
            ),
            math(r"MH-HM=QH-HQ"),
            paragraph("最后一个等号来自 M=Q+I/3，且 I/3 与 H 对易。"),
            subhead("散度指标"),
            math(r"[xx,xy,xz,yx,yy,yz,zx,zy,zz]"),
            math(r"f_x=\partial_x\Pi_{xx}+\partial_y\Pi_{xy}+\partial_z\Pi_{xz}"),
            math(r"f_y=\partial_x\Pi_{yx}+\partial_y\Pi_{yy}+\partial_z\Pi_{yz}"),
            math(r"f_z=\partial_x\Pi_{zx}+\partial_y\Pi_{zy}+\partial_z\Pi_{zz}"),
        ],
    ),
    (
        "当前实现核验结果与参数映射",
        [
            bullet("应力使用 raw H；只有 Q 演化系数除以 γ。"),
            bullet("active stress 只进入总应力一次；active tangential force 仅用于诊断。"),
            bullet("distortion stress 按 DCT/DST 的 z 奇偶性分开求散度。"),
            bullet("Stokes 算子 A=fric+ηk²，对应 (fric−η∇²)u+∇p=f，符号正确。"),
            bullet("分子场、应力、能量配对、Q RHS、adapter 和线性算子共 8 项测试通过。"),
            bullet("另以 20 组随机不可压缩张量验证 Plane 旧展开式与共享 adapter 一致。"),
            subhead("Frank–Landau–de Gennes 映射"),
            math(r"K=2L_1q_{\rm eq}^2,\qquad L_1=\frac{K}{2q_{\rm eq}^2}"),
            math(r"S_{\rm eq}=\frac{1}{3},\quad q_{\rm eq}=\frac{1}{2},\quad L_1=2K"),
            subhead("活性数"),
            math(r"\mathcal{A}=H_{\rm ch}\sqrt{\frac{\zeta}{K}}"),
            paragraph(
                "这里 H_ch 是通道高度，H 是分子场；必须分开记号。当前支持的"
                " Plane_beris_edwards_stokes.py 已使用共享模型，并记录实现"
                " provenance 与配置哈希。"
            ),
        ],
    ),
    (
        "当前模型问题总览",
        [
            paragraph("应力闭合目前正确；以下问题决定它是否能严格复现 Shendruk。"),
            bullet("准静态 Stokes 删除了论文中的惯性、速度瞬态和平均动量演化。"),
            bullet("fric=0 + zero_mean 会投影掉非零切向平均 nematic force。"),
            bullet("friction 模式虽解除零模，却引入论文 bulk 模型没有的 Brinkman drag。"),
            bullet("当前 free-slip 是运动学条件，不是总 nematic+viscous 牵引为零。"),
            bullet("Q 的 Neumann/free anchoring 只对应 Fig. 4 的一个分支。"),
            bullet("仅实现 L2=L3=0，不能覆盖 twist/bend/splay 各向异性。"),
            bullet("谱方法、初始化和时间推进与论文的 hybrid LB/FD 不相同。"),
            bullet("8 项测试覆盖张量、Q RHS 和 adapter，但缺完整混合基 Stokes manufactured test。"),
            bullet("共享 helper 已接入当前 Plane_beris_edwards_stokes.py；中间参考脚本仍保留旧展开。"),
            bullet("保存的 p 是吸收各向同性应力后的有效压力。"),
            paragraph("下面给出最重要三个问题的数学原因。"),
        ],
    ),
    (
        "问题 1：准静态近似与平均动量",
        [
            subhead("论文与代码的方程不同"),
            math(
                r"\rho(\partial_t+\mathbf{u}\!\cdot\!\nabla)\mathbf{u}"
                r"=\nabla\!\cdot\Pi\qquad\mathrm{(Shendruk)}"
            ),
            math(
                r"0=-\nabla p+\eta\nabla^2\mathbf{u}-\mathrm{fric}\,\mathbf{u}"
                r"+\nabla\!\cdot\Pi^{\rm nem}\qquad\mathrm{(current)}",
                12.0,
            ),
            paragraph(
                "即使 Reynolds 数预计很小，也需要与惯性参考解或 Re 收敛对比；"
                "不能仅凭形式判断临界活性数不受影响。"
            ),
            subhead("fric=0 时的切向零模可解性"),
            math(r"\langle f_x\rangle=\langle f_y\rangle=0\quad\mathrm{(steady\ solvability)}"),
            math(
                r"\langle f_x\rangle=\frac{1}{V}\int_{\partial V}"
                r"\Pi^{\rm nem}_{xj}n_j,\mathrm{d}A"
            ),
            math(
                r"=\frac{1}{H_{\rm ch}}\left[\langle\Pi^{\rm nem}_{xz}\rangle_{xy}"
                r"\right]_{z=0}^{z=H_{\rm ch}}"
            ),
            paragraph(
                "Neumann Q 边界不保证这个表面差为零。zero_mean 将相应 RHS 零模"
                "投影掉，等价于加入均匀反力，不是压力 gauge。"
            ),
            math(
                r"\mathrm{fric}\,\langle u_{x,y}\rangle=\langle f_{x,y}\rangle"
                r"\qquad\mathrm{(friction\ mode)}"
            ),
        ],
    ),
    (
        "问题 2：free-slip 与 anchoring",
        [
            subhead("当前运动学边界"),
            math(r"u_z=0,\qquad \partial_z u_x=\partial_z u_y=0"),
            paragraph("它令黏性切向牵引为零，但总切向牵引仍含 nematic 部分："),
            math(
                r"t_x=2\eta E_{xz}+\Pi^{\rm nem}_{xz},\qquad"
                r"t_y=2\eta E_{yz}+\Pi^{\rm nem}_{yz}"
            ),
            paragraph(
                "因此当前边界是 homogeneous kinematic free-slip，不等价于"
                " n_jΠ^tot_ij=0。若需要总 traction-free，速度边界必须与 nematic "
                "应力耦合。"
            ),
            subhead("Q 边界对应关系"),
            math(r"\partial_zQ_{ij}=0\qquad\mathrm{(current\ free\ anchoring)}"),
            paragraph(
                "Shendruk 默认 strong planar anchoring；Fig. 4 另有 free-slip/"
                "free-anchoring 数据。当前组合可对应后者，不能代表所有曲线。"
            ),
        ],
    ),
    (
        "问题 3：弹性范围、数值差异与测试缺口",
        [
            subhead("超过 one-constant 时"),
            math(
                r"f_{\rm el}=\frac{L_1}{2}\partial_kQ_{ij}\partial_kQ_{ij}"
                r"+\frac{L_2}{2}(\partial_kQ_{kj})(\partial_iQ_{ij})"
            ),
            math(
                r"\qquad +\frac{L_3}{2}Q_{ki}(\partial_kQ_{jl})(\partial_iQ_{jl})"
            ),
            math(
                r"\Pi^d_{ij}=-\partial_iQ_{kl}"
                r"\frac{\partial f_{\rm el}}{\partial(\partial_jQ_{kl})}"
            ),
            paragraph(
                "研究 twist/bend/splay 不等时，H 和 Π^d 都要重新变分；不能只把"
                " L1 替换成一个有效常数。"
            ),
            subhead("数值与测试"),
            bullet("论文为 hybrid lattice-Boltzmann/finite difference；当前为 FFT/DCT/DST 谱方法。"),
            bullet("需验证网格、时间步、初始化和去混叠对临界活性数及缺陷统计的影响。"),
            bullet("应增加完整应力散度、Schur 压力、墙面奇偶基和零模的 manufactured tests。"),
            bullet("默认 cubic_half 去掉最高 DST 边缘模；dealias=none 仍需单独测试。"),
            subhead("共享 helper 与实际 Plane 路径"),
            bullet("当前支持的 Plane_beris_edwards_stokes.py 已直接使用共享 Q adapter 与应力 helper。"),
            bullet("20 组随机张量验证了中间脚本旧展开与共享实现一致；应继续用回归测试和实现哈希防止漂移。输出 p 是有效压力，不能直接和 LB 压力逐点比较。"),
        ],
    ),
    (
        "建议优先级与参考资料",
        [
            bullet("P0：严格复现时保留惯性，或至少显式演化两个切向平均动量。"),
            bullet("P0：明确目标 Fig. 4 曲线，分别实现 strong/free anchoring 与 no/free slip。"),
            bullet("P1：建立 manufactured solutions 和逐项功率/能量预算。"),
            bullet("P1：保持 Plane_beris_edwards_stokes.py 使用共享 adapter/helper，并保留 provenance 与回归测试。"),
            bullet("P2：完成 one-constant 后，再实现 L2、L3 对 H 和形变应力的贡献。"),
            paragraph(
                "若目标仅为自洽的低 Reynolds 数 one-constant 谱模型，当前应力可"
                "保留，但应准确描述为：准静态不可压缩 Stokes–Brinkman、运动学"
                " free-slip，并明确 plug-mode convention。"
            ),
            subhead("主要来源"),
            bullet("Shendruk et al., Phys. Rev. E 98, 010601 (2018), DOI: 10.1103/PhysRevE.98.010601"),
            bullet("Marenduzzo et al., Phys. Rev. E 76, 031921 (2007), DOI: 10.1103/PhysRevE.76.031921"),
            bullet("Beris & Edwards, Thermodynamics of Flowing Systems (1994)."),
            bullet("Shendruk et al., Soft Matter 13, 3853 (2017), DOI: 10.1039/C6SM02310J"),
            subhead("对应代码"),
            paragraph("Plane_beris_edwards_stokes.py；作为中间参考的 Plane_shendruk_stokes.py；pssolver/models/active_nematics/beris_edwards.py；tests/test_beris_edwards_stress.py；scripts_plane/run_beris_edwards_validation.py。"),
            paragraph("可编辑的完整规范源文件：docs/beris_edwards_stokes_derivation_zh.tex"),
        ],
    ),
]


def wrap_lines(text: str, width: int):
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [""]


class DocumentRenderer:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.fig = None
        self.y = 0.0
        self.page = 0
        self.chapter = ""
        self.continuation = False

    def new_page(self, chapter: str, continuation: bool = False):
        if self.fig is not None:
            self.finish_page()
        self.page += 1
        self.chapter = chapter
        self.continuation = continuation
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        title = chapter + ("（续）" if continuation else "")
        self.fig.text(
            0.075,
            0.955,
            title,
            fontproperties=ZH_BOLD,
            fontsize=17,
            color="#17365D",
            va="top",
        )
        self.fig.lines.append(
            plt.Line2D([0.075, 0.925], [0.925, 0.925], transform=self.fig.transFigure, color="#9CB6CE", lw=1.0)
        )
        self.y = 0.895

    def ensure(self, needed: float):
        if self.y - needed < 0.075:
            self.new_page(self.chapter, continuation=True)

    def add(self, block):
        kind = block[0]
        if kind == "h":
            self.ensure(0.07)
            self.fig.text(
                0.08,
                self.y,
                block[1],
                fontproperties=ZH_BOLD,
                fontsize=12.5,
                color="#284B63",
                va="top",
            )
            self.y -= 0.052
            return

        if kind in {"p", "b"}:
            prefix = "• " if kind == "b" else ""
            indent = 0.10 if kind == "b" else 0.08
            width = 48 if kind == "b" else 52
            lines = wrap_lines(prefix + block[1], width)
            needed = 0.0305 * len(lines) + 0.013
            self.ensure(needed)
            self.fig.text(
                indent,
                self.y,
                "\n".join(lines),
                fontproperties=ZH,
                fontsize=10.3,
                linespacing=1.48,
                color="#202124",
                va="top",
            )
            self.y -= needed
            return

        if kind == "m":
            expr, size = block[1], block[2]
            needed = 0.061
            self.ensure(needed)
            self.fig.text(
                0.50,
                self.y - 0.005,
                f"${expr}$",
                fontsize=size,
                color="#111111",
                ha="center",
                va="top",
            )
            self.y -= needed
            return

        raise ValueError(f"Unknown block kind: {kind}")

    def finish_page(self):
        self.fig.text(
            0.075,
            0.035,
            "Beris–Edwards / Shendruk 模型推导与核验",
            fontproperties=ZH,
            fontsize=8.2,
            color="#6B7280",
        )
        self.fig.text(
            0.925,
            0.035,
            str(self.page),
            fontproperties=ZH,
            fontsize=8.2,
            color="#6B7280",
            ha="right",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None


def title_page(pdf: PdfPages):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.5,
        0.76,
        "Beris–Edwards 主动向列相应力\n与 Stokes 方程",
        fontproperties=ZH_BOLD,
        fontsize=26,
        ha="center",
        va="center",
        color="#17365D",
        linespacing=1.45,
    )
    fig.text(
        0.5,
        0.62,
        "推导、代码核验及当前模型问题",
        fontproperties=ZH,
        fontsize=17,
        ha="center",
        color="#284B63",
    )
    fig.text(
        0.5,
        0.50,
        r"$H:\mathcal{S}+\Pi^R:W=0$",
        fontsize=22,
        ha="center",
        color="#111111",
    )
    fig.text(
        0.5,
        0.39,
        "针对 Plane_beris_edwards_stokes.py 与共享模型\n2026 年 8 月 29 日",
        fontproperties=ZH,
        fontsize=12,
        ha="center",
        color="#4B5563",
        linespacing=1.6,
    )
    fig.text(
        0.5,
        0.18,
        "规范源：beris_edwards_stokes_derivation_zh.tex",
        fontproperties=ZH,
        fontsize=9.5,
        ha="center",
        color="#6B7280",
    )
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(OUTPUT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Beris–Edwards 主动向列相应力与 Stokes 方程"
        metadata["Author"] = "Codex mathematical derivation and code audit"
        metadata["Subject"] = "Shendruk/Beris–Edwards stress derivation and current model limitations"
        title_page(pdf)
        renderer = DocumentRenderer(pdf)
        for chapter, blocks in CHAPTERS:
            renderer.new_page(chapter)
            for block in blocks:
                renderer.add(block)
        if renderer.fig is not None:
            renderer.finish_page()
    print(OUTPUT)


if __name__ == "__main__":
    main()
