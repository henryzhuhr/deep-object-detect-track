import sys


class VersionUtils:
    @staticmethod
    def get_python_vesion(level: int = 2) -> str:
        python_versions = [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
        assert_str = f"level should be less than or equal to {python_versions.__len__()}, {python_versions}"
        assert level <= python_versions.__len__(), assert_str
        return "".join([str(p) for p in python_versions[:level]])

    @staticmethod
    def is_v_ge(current_version: str, target_version: str, level: int = -1) -> bool:
        """is current_version greater than or equal to target_version"""
        cvs, tvs = current_version.split("."), target_version.split(".")  # current_version, target_version
        vlen = len(tvs) if level == -1 else level
        for i in range(vlen):
            if int(cvs[i]) > int(tvs[i]):
                return True
        return False

    @staticmethod
    def is_v_eq(current_version: str, target_version: str, level: int = -1) -> bool:
        """is current_version equal to target_version"""
        cvs, tvs = current_version.split("."), target_version.split(".")  # current_version, target_version
        vlen = len(tvs) if level == -1 else level
        for i in range(vlen):
            if int(cvs[i]) != int(tvs[i]):
                return False
        return True
