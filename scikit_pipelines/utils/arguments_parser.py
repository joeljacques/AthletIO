import argparse


class ArgumentsParser(object):

    def __init__(self, prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=None,
                 formatter_class=argparse.HelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=True,
                 allow_abbrev=True):

        if parents is None:
            parents = []
        self.flags = []
        self.__parser = argparse.ArgumentParser(prog=prog,
                                                usage=usage,
                                                description=description,
                                                epilog=epilog,
                                                parents=parents,
                                                formatter_class=formatter_class,
                                                prefix_chars=prefix_chars,
                                                fromfile_prefix_chars=fromfile_prefix_chars,
                                                argument_default=argument_default,
                                                conflict_handler=conflict_handler,
                                                add_help=add_help,
                                                allow_abbrev=allow_abbrev)

    def add_flag(self, flag_name: str, *args, **kwargs):
        if flag_name.replace("-", "") in self.flags:
            raise RuntimeError(f"Flag {flag_name} already exist")
        self.flags.append(flag_name.replace("-", ""))

        self.__parser.add_argument(flag_name, *args, **kwargs)

    def parse(self, args=None, namespace=None):
        return self.__parser.parse_args(args=args, namespace=namespace)


class Flags:
    pass


def get_args_parser(flags: dict, description: str, **kwargs):
    args_parser = ArgumentsParser(description=description, **kwargs)
    for flag in flags.keys():
        if not isinstance(flags[flag], dict):
            raise RuntimeError("Flag value must also be a dictionary")
        args_parser.add_flag(flag, **flags[flag])
    return args_parser


__all__ = ["ArgumentsParser", "Flags", "get_args_parser"]
