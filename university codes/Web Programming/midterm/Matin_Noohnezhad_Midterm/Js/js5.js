var add = JSON.parse(window.sessionStorage.getItem("add"));
if (add == null) {
    add = false;
}
var editIndex = JSON.parse(window.sessionStorage.getItem("editIndex"));
if (editIndex == null) {
    editIndex = 0;
}
var arr = JSON.parse(window.sessionStorage.getItem("array"));
if (arr == null) {
    arr = [
        {fullName: "متین نوح نژاد", phoneNo: "09191235537"},
        {fullName: "عرفان سلیمانی", phoneNo: "09175635837"},
        {fullName: "هومن کیانی", phoneNo: "09617682345"},
        {fullName: "حامد ساحل", phoneNo: "09617790693"}
    ];
}
fillTable();

function fillTable() {
    // window.alert(1);
    var text = "<tr>\n" +
        "            <th>ردیف</th>\n" +
        "            <th>نام و نام خانوادگی</th>\n" +
        "            <th>شماره تماس</th>\n" +
        "            <th>عملیات</th>\n" +
        "        </tr>";
    for (var i = 0; i < arr.length; i++) {
        text += "<tr>\n" +
            "            <th>" + (i + 1).toString() + "</th>\n" +
            "            <th>" + arr[i].fullName + "</th>\n" +
            "            <th>" + arr[i].phoneNo + "</th>\n" +
            "            <th>\n" +
            "                <button id=\"b1\" onclick=\"editRow(" + i.toString() + ")\">ویرایش</button>\n" +
            "                <button id=\"b2\" onclick=\"deleteRow(" + i.toString() + ")\">حذف</button>\n" +
            "            </th>\n" +
            "        </tr>";
    }
    document.getElementById("myTable").innerHTML = text;
}

function editRow(input) {
    add = false;
    editIndex = input;
    window.sessionStorage.setItem("add", JSON.stringify(add));
    window.sessionStorage.setItem("editIndex", JSON.stringify(editIndex));
    window.sessionStorage.setItem("array", JSON.stringify(arr));
    window.open("EnterInformation.html", "_self");
}

function deleteRow(input) {
    arr.splice(input, 1);
    fillTable();
}

function addRow() {
    add = true;
    window.sessionStorage.setItem("add", JSON.stringify(add));
    window.sessionStorage.setItem("editIndex", JSON.stringify(editIndex));
    window.sessionStorage.setItem("array", JSON.stringify(arr));
    window.open("EnterInformation.html", "_self");
}

function formSubmission() {
    var fullName = document.forms["myForm"]["fullName"].value;
    var phoneNo = document.forms["myForm"]["phoneNo"].value;
    if (add) {
        var obj = {fullName: fullName, phoneNo: phoneNo};
        arr.push(obj);
    } else {
        arr[editIndex].fullName = fullName;
        arr[editIndex].phoneNo = phoneNo;
    }
    window.sessionStorage.setItem("add", JSON.stringify(add));
    window.sessionStorage.setItem("editIndex", JSON.stringify(editIndex));
    window.sessionStorage.setItem("array", JSON.stringify(arr));
}

