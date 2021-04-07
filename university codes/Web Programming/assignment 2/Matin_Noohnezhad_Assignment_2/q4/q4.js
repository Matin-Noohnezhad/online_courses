var codeSortAscending = false;
var titleSortAscending = false;
var noUnitSortAscending = false;
var tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
    "    <tr>\n" +
    "        <th onclick=\"sortTable('code')\">کد درس</th>\n" +
    "        <th onclick=\"sortTable('title')\">عنوان درس</th>\n" +
    "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
    "        </th>\n" +
    "        <th>لینک به صفحه درس</th>\n" +
    "    </tr>";
var arr = [
    {
        code: "COM4106",
        title: "ماشین لرنینگ",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=1224"
    },
    {
        code: "LAB3115",
        title: "آزمایشگاه شبکه",
        noUnit: 1,
        linkAdd: "https://course.bihe36.info/course/view.php?id=3206"
    },
    {
        code: "COM4136",
        title: "برنامه نویسی سیستم",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=2546"
    },
    {
        code: "COM4134",
        title: "برنامه نویسی وب",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=1707"
    },
    {
        code: "COR3126",
        title: "فراسوی فرهنگ رقابت و ستیز",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=3241"
    },
    {code: "COM4106", title: "ماشین لرنینگ2", noUnit: 3, linkAdd: "https://course.bihe36.info/course/view.php?id=1224"},
    {
        code: "LAB3115",
        title: "آزمایشگاه شبکه2",
        noUnit: 1,
        linkAdd: "https://course.bihe36.info/course/view.php?id=3206"
    },
    {
        code: "COM4136",
        title: "برنامه نویسی سیستم2",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=2546"
    },
    {
        code: "COM4134",
        title: "برنامه نویسی وب2",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=1707"
    },
    {
        code: "COR3126",
        title: "فراسوی فرهنگ رقابت و ستیز2",
        noUnit: 3,
        linkAdd: "https://course.bihe36.info/course/view.php?id=3241"
    },
];
fillTable();


function fillTable() {
    var text = tableHeader;
    var n = arr.length;
    for (var i = 0; i < n; i++) {
        text += "<tr>\n" +
            "        <td>" + arr[i].code + "</td>\n" +
            "        <td>" + arr[i].title + "</td>\n" +
            "        <td>" + arr[i].noUnit + "</td>\n" +
            "        <td><a href=' " + arr[i].linkAdd + "  '>لینک درس</a></td>\n" +
            "    </tr>";
    }
    document.getElementById("t01").innerHTML = text;
}

function compareCode(a, b) {
    if (a.code < b.code)
        return -1;
    if (a.code > b.code)
        return 1;
    return 0;
}

function compareTitle(a, b) {
    if (a.title < b.title)
        return -1;
    if (a.title > b.title)
        return 1;
    return 0;
}

function compareNoUnit(a, b) {
    if (a.noUnit < b.noUnit)
        return -1;
    if (a.noUnit > b.noUnit)
        return 1;
    return 0;
}

function sortTable(col) {
    if (col == "code") {
        arr = arr.sort(compareCode);
        if (codeSortAscending) {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس\n" +
                "            <img src=\"ascending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            arr = arr.reverse();
            codeSortAscending = false;
        } else {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس\n" +
                "            <img src=\"descending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            codeSortAscending = true;
        }
        fillTable();
    } else if (col == "title") {
        arr = arr.sort(compareTitle);
        if (titleSortAscending) {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس\n" +
                "            <img src=\"ascending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            arr = arr.reverse();
            titleSortAscending = false;
        } else {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس\n" +
                "            <img src=\"descending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            titleSortAscending = true;
        }
        fillTable();
    } else if (col == "noUnit") {
        arr = arr.sort(compareNoUnit);
        if (noUnitSortAscending) {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس</th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس</th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "            <img src=\"ascending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            arr = arr.reverse();
            noUnitSortAscending = false;
        } else {
            tableHeader = "    <caption><b>دروس دانشجو</b></caption>\n" +
                "    <tr>\n" +
                "        <th onclick=\"sortTable('code')\">کد درس</th>\n" +
                "        <th onclick=\"sortTable('title')\">عنوان درس</th>\n" +
                "        <th onclick=\"sortTable('noUnit')\">تعداد واحد\n" +
                "            <img src=\"descending_sign.png\" style=\"width: 4%;height: auto;display: inline;\">\n" +
                "        </th>\n" +
                "        <th>لینک به صفحه درس</th>\n" +
                "    </tr>";
            noUnitSortAscending = true;
        }
        fillTable();
    }
}
