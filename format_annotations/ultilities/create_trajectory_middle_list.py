__all__ = [
	"object_type_name",
	"object_type_id",
	"object_warehouse_018",
]

object_type_name = {
	0 : "Person", # green
	1 : "Forklift", # green
	2 : "NovaCarter", # pink
	3 : "Transporter", # yellow
	4 : "FourierGR1T2", # purple
	5 : "AgilityDigit", # blue
}

object_type_id = {
	"Person"       : 0, # green
	"Forklift"     : 1, # green
	"NovaCarter"   : 2, # pink
	"Transporter"  : 3, # yellow
	"FourierGR1T2" : 4, # purple
	"AgilityDigit" : 5, # blue
}

color_chart = {
	"Person"      : (77, 109, 163), # brown
	"Forklift"    : (162, 245, 214), # light yellow
	"NovaCarter"  : (245, 245, 245), # light pink
	"Transporter" : (0  , 255, 255), # yellow
	"FourierGR1T2": (164, 17 , 157), # purple
	"AgilityDigit": (235, 229, 52) , # blue
}


object_warehouse_017 = {
	"name"       : "Warehouse_017",
	"start_frame": 0,
	"end_frame"  : 8999, # 9000 frames
	"interval"   : 30,
	"objects"    : [
		{
			"id"    : 1,
			"type"  : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 2,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 3,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 4,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 5,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 6,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 7,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 8,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 9,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 10,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 11,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 12,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 13,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 14,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 15,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 16,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 17,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 18,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 19,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 20,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		}
	]
}


object_warehouse_018 = {
	"name"       : "Warehouse_018",
	"start_frame": 0,
	"end_frame"  : 8999, # 9000 frames
	"interval"   : 30,
	"objects"    : [
		{
			"id"    : 1,
			"type"  : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1  , "end": 30},
				{"type": "line" , "start": 30 , "end": 480},
				{"type": "clock", "start": 480, "end": 585},
			]
		},
		{
			"id"   : 2,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 3,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 4,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 5,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 6,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 7,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 8,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 9,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 10,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 11,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 12,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 13,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 14,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 15,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 16,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 17,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 18,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 19,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 20,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 21,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
	]
}


object_warehouse_019 = {
	"name"       : "Warehouse_019",
	"start_frame": 0,
	"end_frame"  : 8999, # 9000 frames
	"interval"   : 30,
	"objects"    : [
		{
			"id"    : 1,
			"type"  : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				# {"type": "stop"         , "start": 1   , "end": 30},
				# {"type": "stop"         , "start": 30  , "end": 90},
				# {"type": "line"         , "start": 90  , "end": 270},
				# {"type": "stop"         , "start": 270  , "end": 360},
				# {"type": "line"         , "start": 360  , "end": 480},
				# {"type": "stop"         , "start": 480  , "end": 765},
			]
		},
		{
			"id"   : 2,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop"         , "start": 1   , "end": 30},
				{"type": "clock"        , "start": 30  , "end": 360},
				{"type": "line"         , "start": 360 , "end": 1320},
				{"type": "counter_clock", "start": 1320, "end": 1725},
				{"type": "line"         , "start": 1725, "end": 2310},
				{"type": "counter_clock", "start": 2310, "end": 2505},
				{"type": "line"         , "start": 2505, "end": 3105},
				{"type": "clock"        , "start": 3105, "end": 3285},
				{"type": "line"         , "start": 3285, "end": 3885},
				{"type": "counter_clock", "start": 3885, "end": 4080},
				{"type": "line"         , "start": 4080, "end": 4680},
				{"type": "clock"        , "start": 4680, "end": 4875},
				{"type": "line"         , "start": 4875, "end": 5475},
				{"type": "counter_clock", "start": 5475, "end": 5820},
				{"type": "line"         , "start": 5820, "end": 6420},
				{"type": "clock"        , "start": 6420, "end": 6615},
				{"type": "line"         , "start": 6615, "end": 7200},
				{"type": "counter_clock", "start": 7200, "end": 7395},
				{"type": "line"         , "start": 7395, "end": 7995},
				{"type": "clock"        , "start": 7995, "end": 8190},
				{"type": "line"         , "start": 8190, "end": 8790},
				{"type": "counter_clock", "start": 8190, "end": 8999},
			]
		},
		{
			"id"   : 3,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 4,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 5,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 6,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 7,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 8,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop"         , "start": 1   , "end": 30},
				{"type": "clock"        , "start": 30  , "end": 165},
				{"type": "line"         , "start": 165 , "end": 720},
				{"type": "counter_clock", "start": 720 , "end": 810},
				{"type": "line"         , "start": 810 , "end": 825},
				{"type": "stop"         , "start": 825 , "end": 1035},
				{"type": "counter_clock", "start": 1035, "end": 1110},
				{"type": "line"         , "start": 1110, "end": 1950},
				{"type": "clock"        , "start": 1950, "end": 2145},
				{"type": "link"         , "start": 2145, "end": 2985},
				{"type": "counter_clock", "start": 2985, "end": 3180},
				{"type": "line"         , "start": 3180, "end": 4020},
				{"type": "clock"        , "start": 4020, "end": 4215},
				{"type": "line"         , "start": 4215, "end": 5055},
				{"type": "counter_clock", "start": 5055, "end": 5250},
				{"type": "line"         , "start": 5250, "end": 6090},
				{"type": "clock"        , "start": 6090, "end": 6435},
				{"type": "line"         , "start": 6435, "end": 7275},
				{"type": "counter_clock", "start": 7275, "end": 7470},
				{"type": "line"         , "start": 7470, "end": 8310},
				{"type": "clock"        , "start": 8310, "end": 8505},
				{"type": "line"         , "start": 8505, "end": 8999},
			]
		},
		{
			"id"   : 9,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 10,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 11,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 12,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 13,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1   , "end": 30},
				{"type": "clock", "start": 30  , "end": 345},
				{"type": "line" , "start": 345 , "end": 900},
				{"type": "clock", "start": 900 , "end": 1080},
				{"type": "line" , "start": 1080, "end": 1980},
				{"type": "line" , "start": 1980, "end": 2145},
				{"type": "line" , "start": 2145, "end": 2430},
				{"type": "clock", "start": 2430, "end": 2820},
				{"type": "clock", "start": 2820, "end": 3165},
				{"type": "line" , "start": 3165, "end": 3255},
				{"type": "line" , "start": 3255, "end": 4065},
				{"type": "clock", "start": 4065, "end": 4275},
				{"type": "line" , "start": 4275, "end": 5160},
				{"type": "line" , "start": 5160, "end": 5325},
				{"type": "line" , "start": 5325, "end": 5625},
				{"type": "clock", "start": 5625, "end": 5820},
				{"type": "line" , "start": 5820, "end": 6165},
				{"type": "line" , "start": 6165, "end": 6195},
				{"type": "line" , "start": 6195, "end": 7080},
				{"type": "clock", "start": 7080, "end": 7275},
				{"type": "line", "start": 7275, "end": 8175},
				{"type": "line", "start": 8175, "end": 8325},
				{"type": "line", "start": 8325, "end": 8625},
				{"type": "clock", "start": 8325, "end": 8999},
			]
		},
		{
			"id"   : 14,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1   , "end": 30},
				{"type": "stop" , "start": 30  , "end": 195},
				{"type": "line" , "start": 195 , "end": 915},
				{"type": "clock", "start": 915 , "end": 975},
				{"type": "line" , "start": 975 , "end": 1095},
				{"type": "line" , "start": 1095, "end": 1320},
				{"type": "line" , "start": 1320, "end": 1590},
				{"type": "line" , "start": 1590, "end": 1710},
			]
		},
		{
			"id"   : 15,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 16,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 17,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 18,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 19,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1   , "end": 30},
				{"type": "stop" , "start": 30  , "end": 120},
				{"type": "line" , "start": 120 , "end": 570},
				{"type": "line" , "start": 570 , "end": 630},
				{"type": "clock", "start": 630 , "end": 765},
				{"type": "line" , "start": 765 , "end": 1845},
				{"type": "clock", "start": 1845, "end": 2190},
				{"type": "line" , "start": 2190, "end": 3270},
				{"type": "clock", "start": 3270, "end": 3435},
				{"type": "line" , "start": 3435, "end": 4530},
				{"type": "clock", "start": 4530, "end": 4710},
				{"type": "line" , "start": 4710, "end": 5790},
				{"type": "clock", "start": 5790, "end": 5970},
				{"type": "line" , "start": 5970, "end": 7050},
				{"type": "clock", "start": 7050, "end": 7500},
				{"type": "line" , "start": 7500, "end": 8580},
				{"type": "clock", "start": 8580, "end": 8940},
				{"type": "line" , "start": 8940, "end": 8999},
			]
		},
		{
			"id"   : 20,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 21,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 22,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 23,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 24,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 25,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 26,
			"type" : "Forklift",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1   , "end": 8999},
			]
		},
		{
			"id"   : 27,
			"type" : "Forklift",
			"shape" : [0,0,0], # width, height, length
			"periods": [
				{"type": "stop" , "start": 1   , "end": 8999},
			]
		},
	]
}


object_warehouse_020 = {
	"name"       : "Warehouse_020",
	"start_frame": 0,
	"end_frame"  : 8999, # 9000 frames
	"interval"   : 30,
	"objects"    : [
		{
			"id"    : 1,
			"type"  : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 2,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 3,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 4,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 5,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 6,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 7,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 8,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 9,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 10,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 11,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 12,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 13,
			"type" : "NovaCarter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 14,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 15,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 16,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 17,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 18,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 19,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 20,
			"type" : "FourierGR1T2",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 21,
			"type" : "AgilityDigit",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 22,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 23,
			"type" : "Person",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 24,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 25,
			"type" : "Transporter",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 26,
			"type" : "Forklift",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 27,
			"type" : "Forklift",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
		{
			"id"   : 28,
			"type" : "Forklift",
			"shape" : [0,0,0], # width, height, length
			"periods": [
			]
		},
	]
}


