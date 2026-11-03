# S-19 公开执行声明及 venv 引导更正

安装版 CPU 演练发现 legacy Channel 的声明是 ExecutionSpec，而非
TorchSpectralExecution。支持适配器分别处理两类声明；六条路径的回归测试
确认只改变显式 device，原声明和其他元数据不变。未改变运行时或 frozen helper。

新 venv 的 ensurepip 版本会遮住基线：必须只在这三个新环境中安装 C 的
pip 25.2 和 setuptools 78.1.1，并把两个 wheel 绑定到部署清单。
先前停止日志保留，不改写为成功。

明确导入边界：未改动的 Consumer conftest 会把 Provider 参考目录放入
sys.path；两个已安装包先加载，测试后逐个核对所有模块确实来自当前 venv。
不把“未出现源码遮蔽”误写成“conftest 从未修改 sys.path”。
