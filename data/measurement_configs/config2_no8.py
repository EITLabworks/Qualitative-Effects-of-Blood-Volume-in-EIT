"""
Project ：quali_effects_eit_measurements
Directory: data/measurement_configs
File : config2_no8.py
Author ：Patricia Fuchs
Date ：14.01.2026 10:38
"""
from data.loading_data import MeaMeta
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

input_folders=["Phase2Cylinder/", "Phase3CylinderBallon/","Phase4Tubes/", "Phase5TubesBallon/" ]
probetypes = ["none", "ballon", "smallcylinder", "mediumcylinder", "largecylinder", "smalltube", "mediumtube", "largetube"]
positions = [["none", "none"],[0,0], [2,200], [2,230], [4,200], [4,230], [6,200], [6,215], [6,230]]
vents= ["none", "ballon"]
mealist= ["static", "high", "low"]          # not timedependent


testcond = []
testcond.append(MeaMeta("", "smallcylinder", [4,200], "none", "static"))
testcond.append(MeaMeta("", "smallcylinder", [2,200], "ballon", "static"))
testcond.append(MeaMeta("", "smallcylinder", [6,215], "ballon", "static"))
testcond.append(MeaMeta("", "mediumcylinder", [2,230], "none", "static"))
testcond.append(MeaMeta("", "mediumcylinder", [4,200], "ballon", "static"))
testcond.append(MeaMeta("", "largecylinder", [0,0], "none", "static"))
testcond.append(MeaMeta("", "largecylinder", [6,200], "none", "static"))

testcond.append(MeaMeta("", "smalltube", [4,230], "none", "low"))
testcond.append(MeaMeta("", "smalltube", [0,0], "ballon", "low"))
testcond.append(MeaMeta("", "mediumtube", [2,200], "none", "low"))
testcond.append(MeaMeta("", "mediumtube", [4,230], "ballon", "low"))
testcond.append(MeaMeta("", "mediumtube", [6,200], "ballon", "low")) ##geändert wegen falsch
testcond.append(MeaMeta("", "largetube", [0,0], "none", "low"))
testcond.append(MeaMeta("", "largetube", [4,230], "ballon", "low"))

testcond.append(MeaMeta("", "smalltube", [0,0], "none", "high"))
testcond.append(MeaMeta("", "smalltube", [2,200], "ballon", "high"))
testcond.append(MeaMeta("", "mediumtube", [2,230], "none", "high"))
testcond.append(MeaMeta("", "mediumtube", [4,200], "ballon", "high"))
testcond.append(MeaMeta("", "largetube", [4,230], "none", "high"))
testcond.append(MeaMeta("", "largetube", [6,200], "none", "high"))
testcond.append(MeaMeta("", "largetube", [6,215], "ballon", "high"))

valicond = []
valicond.append(MeaMeta("", "smallcylinder", [2,200], "none", "static"))
valicond.append(MeaMeta("", "mediumcylinder", [0,0], "none", "static"))
valicond.append(MeaMeta("", "mediumcylinder", [6,215], "ballon", "static"))
valicond.append(MeaMeta("", "largecylinder", [4,230], "ballon", "static"))

valicond.append(MeaMeta("", "smalltube", [2,230], "none", "low"))
valicond.append(MeaMeta("", "mediumtube", [6,230], "none", "low"))
valicond.append(MeaMeta("", "largetube", [2,200], "none", "low"))
valicond.append(MeaMeta("", "largetube", [6,215], "ballon", "low"))

valicond.append(MeaMeta("", "smalltube", [6,230], "none", "high"))
valicond.append(MeaMeta("", "smalltube", [4,200], "ballon", "high"))
valicond.append(MeaMeta("", "mediumtube", [4,200], "none", "high"))
valicond.append(MeaMeta("", "largetube", [2,200], "none", "high"))


