# 用於提升日誌可讀性的顏色代碼
class Colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def log_info(message: str, color: str = Colors.CYAN):
    """以指定顏色輸出一般資訊。"""
    print(f"{color}ℹ️  {message}{Colors.END}")


def log_success(message: str):
    """以綠色輸出成功訊息。"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def log_error(message: str):
    """以紅色輸出錯誤訊息。"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def log_warning(message: str):
    """以黃色輸出警告訊息。"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def log_header(message: str):
    """以醒目格式輸出標題訊息。"""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}🚀 {message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.PURPLE}{'='*60}{Colors.END}\n")
