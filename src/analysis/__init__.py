from .behavior import BehaviorAnalyzer

# Lazy import for modules requiring cv2
def __getattr__(name):
    if name == "Visualizer":
        from .visualizer import Visualizer
        return Visualizer
    if name == "generate_comprehensive_report":
        from .comprehensive_report import generate_comprehensive_report
        return generate_comprehensive_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
