class FinalDataChecker(object):
    def __init__(self):
        self.flag = True
        self.res = dict()

    def check(self, paths):
        # init
        self.flag = True
        self.res = dict()

        if isinstance(paths, dict):
            for key, value in paths.items():
                self.res[key] = dict()
                self.res[key]["path"] = value
        elif isinstance(paths, list):
            for path in paths:
                self.res[path] = dict()
                self.res[path]["path"] = path

        for key in self.res.keys():
            with open(self.res[key]["path"], "r") as fr:
                lines = fr.readlines()
                file_size = len(lines)
                self.res[key]["size"] = file_size
                fr.close()

        keys = list(self.res.keys())
        for i in range(len(keys)):
            if self.res[keys[i]]["size"] != self.res[keys[0]]["size"]:
                self.flag = False

        for key, value in self.res.keys():
            print(key, ":", value["size"])

        if self.flag:
            print("All feature file has the same size...")
        else:
            print("Feature files do not have the same number of data entries!")
